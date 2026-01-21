import math
import torch
from torch import nn
import torch.nn.functional as F
from mamba_ssm import Mamba


class SpectralAttention(nn.Module):
    """Attention mechanism for spectral/channel relationships"""
    
    def __init__(self, channels, reduction_ratio=4):
        super().__init__()
        reduced_channels = max(channels // reduction_ratio, 1)
        
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.SiLU(),
            nn.Linear(reduced_channels, channels, bias=False)
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            channel_attention: [B, C, 1, 1]
        """
        B, C, H, W = x.shape
        
        # Average and max pooling
        avg_pool = self.avg_pool(x).view(B, C)  # [B, C]
        max_pool = self.max_pool(x).view(B, C)  # [B, C]
        
        # Compute attention weights
        avg_attn = self.fc(avg_pool)  # [B, C]
        max_attn = self.fc(max_pool)  # [B, C]
        
        # Combine and normalize
        channel_attn = torch.sigmoid(avg_attn + max_attn).view(B, C, 1, 1)
        
        return channel_attn


class SpectralSelfAttention(nn.Module):
    """Multi-head self-attention across spectral tokens"""
    
    def __init__(self, group_channel_num, token_num, num_heads=2):
        super().__init__()
        self.token_num = token_num
        self.group_channel_num = group_channel_num
        self.num_heads = num_heads
        self.head_dim = group_channel_num // num_heads
        
        assert group_channel_num % num_heads == 0, "group_channel_num must be divisible by num_heads"
        
        self.q_proj = nn.Linear(group_channel_num, group_channel_num)
        self.k_proj = nn.Linear(group_channel_num, group_channel_num)
        self.v_proj = nn.Linear(group_channel_num, group_channel_num)
        self.out_proj = nn.Linear(group_channel_num, group_channel_num)
        
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x):
        """
        Args:
            x: [B*H*W, token_num, group_channel_num]
        Returns:
            attended: [B*H*W, token_num, group_channel_num]
        """
        BHW, T, C = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(BHW, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(BHW, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(BHW, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(BHW, T, C)
        
        # Output projection
        out = self.out_proj(out)
        
        return out


class CrossSpectralSpatialAttention(nn.Module):
    """Cross-attention between spectral tokens and spatial context"""
    
    def __init__(self, group_channel_num, token_num, num_heads=2):
        super().__init__()
        self.token_num = token_num
        self.group_channel_num = group_channel_num
        self.num_heads = num_heads
        self.head_dim = group_channel_num // num_heads
        
        assert group_channel_num % num_heads == 0
        
        # Queries from spectral tokens, Keys/Values from spatial context
        self.q_proj = nn.Linear(group_channel_num, group_channel_num)
        self.k_proj = nn.Linear(group_channel_num, group_channel_num)
        self.v_proj = nn.Linear(group_channel_num, group_channel_num)
        self.out_proj = nn.Linear(group_channel_num, group_channel_num)
        
        self.scale = self.head_dim ** -0.5
        
    def forward(self, spectral_tokens, spatial_context):
        """
        Args:
            spectral_tokens: [B*H*W, token_num, group_channel_num]
            spatial_context: [B*H*W, token_num, group_channel_num] - same tokens but from different positions
        Returns:
            attended: [B*H*W, token_num, group_channel_num]
        """
        BHW, T, C = spectral_tokens.shape
        
        q = self.q_proj(spectral_tokens).view(BHW, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(spatial_context).view(BHW, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(spatial_context).view(BHW, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(BHW, T, C)
        out = self.out_proj(out)
        
        return out


class SpeMambaWithAttention(nn.Module):
    """Enhanced SpeMamba with spectral attention mechanisms"""
    
    def __init__(self, channels, token_num=8, use_residual=True, group_num=4,
                 attention_type='channel', num_heads=2, use_self_attention=True):
        """
        Args:
            attention_type: 'channel', 'self', 'cross', or 'hybrid'
                - 'channel': Channel-wise attention (CBAM-style)
                - 'self': Self-attention across spectral tokens
                - 'cross': Cross-attention with spatial context
                - 'hybrid': Combine multiple attention types
            num_heads: Number of attention heads for self/cross attention
            use_self_attention: Whether to use self-attention in hybrid mode
        """
        super(SpeMambaWithAttention, self).__init__()
        self.token_num = token_num
        self.use_residual = use_residual
        self.attention_type = attention_type

        self.group_channel_num = math.ceil(channels / token_num)
        self.channel_num = self.token_num * self.group_channel_num

        # Mamba SSM
        self.mamba = Mamba(
            d_model=self.group_channel_num,
            d_state=16,
            d_conv=4,
            expand=2,
        )

        # Attention modules based on type
        if attention_type == 'channel':
            self.channel_attention = SpectralAttention(self.channel_num, reduction_ratio=4)
        elif attention_type == 'self':
            self.self_attention = SpectralSelfAttention(self.group_channel_num, token_num, num_heads)
        elif attention_type == 'cross':
            self.cross_attention = CrossSpectralSpatialAttention(self.group_channel_num, token_num, num_heads)
        elif attention_type == 'hybrid':
            self.channel_attention = SpectralAttention(self.channel_num, reduction_ratio=4)
            if use_self_attention:
                self.self_attention = SpectralSelfAttention(self.group_channel_num, token_num, num_heads)
            # Learnable fusion weights
            num_attn_types = 2 if use_self_attention else 1
            self.attn_fusion_weights = nn.Parameter(torch.ones(num_attn_types + 1) / (num_attn_types + 1))

        self.proj = nn.Sequential(
            nn.GroupNorm(group_num, self.channel_num),
            nn.SiLU()
        )

    def padding_feature(self, x):
        B, C, H, W = x.shape
        if C < self.channel_num:
            pad_c = self.channel_num - C
            pad_features = torch.zeros((B, pad_c, H, W)).to(x.device)
            cat_features = torch.cat([x, pad_features], dim=1)
            return cat_features
        else:
            return x

    def forward(self, x):
        B, C, H, W = x.shape
        x_pad = self.padding_feature(x)
        
        # Apply channel attention before token processing (if applicable)
        if self.attention_type in ['channel', 'hybrid']:
            channel_attn = self.channel_attention(x_pad)
            x_attended = x_pad * channel_attn
        else:
            x_attended = x_pad
        
        # Prepare for token processing
        x_attended = x_attended.permute(0, 2, 3, 1).contiguous()
        B, H, W, C_pad = x_attended.shape
        x_flat = x_attended.view(B * H * W, self.token_num, self.group_channel_num)
        
        # Apply self-attention across spectral tokens (if applicable)
        if self.attention_type == 'self':
            x_flat = x_flat + self.self_attention(x_flat)
            x_flat = self.mamba(x_flat)
        elif self.attention_type == 'hybrid' and hasattr(self, 'self_attention'):
            # Hybrid: combine Mamba output with self-attention
            mamba_out = self.mamba(x_flat)
            self_attn_out = self.self_attention(x_flat)
            
            # Weighted fusion
            weights = F.softmax(self.attn_fusion_weights, dim=0)
            x_flat = weights[0] * x_flat + weights[1] * mamba_out + weights[2] * self_attn_out
        else:
            # Standard Mamba processing
            x_flat = self.mamba(x_flat)
        
        # Reshape back
        x_recon = x_flat.view(B, H, W, C_pad)
        x_recon = x_recon.permute(0, 3, 1, 2).contiguous()
        x_proj = self.proj(x_recon)
        
        if self.use_residual:
            return x + x_proj[:, :C]  # Trim padding channels
        else:
            return x_proj[:, :C]


class LightweightSpeMamba(nn.Module):
    """Lightweight version with efficient attention for low VRAM"""
    
    def __init__(self, channels, token_num=8, use_residual=True, group_num=4):
        super(LightweightSpeMamba, self).__init__()
        self.token_num = token_num
        self.use_residual = use_residual

        self.group_channel_num = math.ceil(channels / token_num)
        self.channel_num = self.token_num * self.group_channel_num

        # Lightweight channel attention (squeeze-excitation style)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.channel_num, self.channel_num // 4, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(self.channel_num // 4, self.channel_num, 1, bias=False),
            nn.Sigmoid()
        )

        self.mamba = Mamba(
            d_model=self.group_channel_num,
            d_state=16,
            d_conv=4,
            expand=2,
        )

        # Lightweight token attention (1x1 conv)
        self.token_attention = nn.Conv1d(self.token_num, self.token_num, 1, groups=1, bias=False)

        self.proj = nn.Sequential(
            nn.GroupNorm(group_num, self.channel_num),
            nn.SiLU()
        )

    def padding_feature(self, x):
        B, C, H, W = x.shape
        if C < self.channel_num:
            pad_c = self.channel_num - C
            pad_features = torch.zeros((B, pad_c, H, W)).to(x.device)
            cat_features = torch.cat([x, pad_features], dim=1)
            return cat_features
        else:
            return x

    def forward(self, x):
        B, C, H, W = x.shape
        x_pad = self.padding_feature(x)
        
        # Channel attention
        channel_weights = self.channel_gate(x_pad)
        x_attended = x_pad * channel_weights
        
        # Token processing
        x_attended = x_attended.permute(0, 2, 3, 1).contiguous()
        B, H, W, C_pad = x_attended.shape
        x_flat = x_attended.view(B * H * W, self.token_num, self.group_channel_num)
        
        # Token attention (lightweight)
        token_weights = torch.sigmoid(self.token_attention(x_flat.transpose(1, 2))).transpose(1, 2)
        x_flat = x_flat * token_weights
        
        # Mamba processing
        x_flat = self.mamba(x_flat)
        
        # Reshape back
        x_recon = x_flat.view(B, H, W, C_pad)
        x_recon = x_recon.permute(0, 3, 1, 2).contiguous()
        x_proj = self.proj(x_recon)
        
        if self.use_residual:
            return x + x_proj[:, :C]
        else:
            return x_proj[:, :C]


# Example usage:
"""
# Option 1: Full attention (choose attention type)
spe_module = SpeMambaWithAttention(
    channels=128,
    token_num=8,
    attention_type='hybrid',  # 'channel', 'self', 'cross', or 'hybrid'
    num_heads=2,
    use_self_attention=True
)

# Option 2: Lightweight version (lower VRAM)
spe_module = LightweightSpeMamba(
    channels=128,
    token_num=8,
    use_residual=True,
    group_num=4
)

# In BothMamba class, replace:
# self.spe_mamba = SpeMamba(...) 
# with:
# self.spe_mamba = SpeMambaWithAttention(...) or LightweightSpeMamba(...)
"""