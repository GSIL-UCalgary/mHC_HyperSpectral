import math
import torch
import torch.nn as nn
from einops import rearrange, einsum
from mamba_ssm import Mamba
import pdb
from random import randrange
# ==================== mHC CORE ====================

class SpaMamba(nn.Module):
    def __init__(self,channels,use_residual=True,group_num=4,use_proj=True):
        super(SpaMamba, self).__init__()
        self.use_residual = use_residual
        self.use_proj = use_proj
        self.mamba = Mamba(  # This module uses roughly 3 * expand * d_model^2 parameters
                           d_model=channels,  # Model dimension d_model
                           d_state=16,  # SSM state expansion factor
                           d_conv=4,  # Local convolution width
                           expand=2,  # Block expansion factor
                           )
        if self.use_proj:
            self.proj = nn.Sequential(
                nn.GroupNorm(group_num, channels),
                nn.SiLU()
            )

    def forward(self,x):
        B = x.shape[0]
        C = x.shape[-1]
        x_re = x.reshape(B, 145, 145, C)
        # x_re = x.permute(0, 2, 3, 1).contiguous()
        B,H,W,C = x_re.shape
        x_flat = x_re.view(1,-1, C)
        x_flat = self.mamba(x_flat)

        x_recon = x_flat.view(B, H, W, C)
        x_recon = x_recon.permute(0, 3, 1, 2).contiguous()
        if self.use_proj:
            x_recon = self.proj(x_recon)
        if self.use_residual:
            return x_recon.reshape(B, -1, C)
        else:
            return x_recon

def sinkhorn_ds_batched(logits, iters=10, tau=0.5, eps=1e-8):
    """
    logits: (N, S, S)
    return: (N, S, S) approx doubly-stochastic
    """
    z = logits / tau
    z = z - z.amax(dim=(-2, -1), keepdim=True)   # numerical stability
    P = torch.exp(z).clamp_min(eps)

    for _ in range(iters):
        P = P / (P.sum(dim=-1, keepdim=True) + eps)
        P = P / (P.sum(dim=-2, keepdim=True) + eps)
    return P



class LocalMHCGenerator(nn.Module):
    """
    Given R(n,s,d), generate:
      H_pre(n):  (1,S)
      H_post(n): (1,S)
      H_res(n):  (S,S)
    """
    def __init__(self, S, D, hidden=128):
        super().__init__()
        self.S = S
        self.D = D
        in_dim = S * D

        self.norm = nn.LayerNorm(in_dim)

        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        self.fc_pre  = nn.Linear(hidden, S)
        self.fc_post = nn.Linear(hidden, S)
        self.fc_res  = nn.Linear(hidden, S * S)

        # gentle init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, R):
        # R: (N, S, D)
        N, S, D = R.shape
        x = R.reshape(N, S * D)
        x = self.norm(x)
        h = self.fc(x)

        H_pre  = self.fc_pre(h).unsqueeze(1)          # (N,1,S)
        H_post = self.fc_post(h).unsqueeze(1)         # (N,1,S)
        H_res  = self.fc_res(h).view(N, S, S)         # (N,S,S)

        return H_pre, H_post, H_res


class SpectralStreamHyperConnections(nn.Module):
    """
    Local / Dynamic mHC
    Spectral tokens = streams
    Spatial branch  = Mamba
    """

    def __init__(
        self,
        channels,
        token_num=4,
        use_residual=True,
        group_num=4,
        mhc_iters=10,
        mhc_tau=0.5,
    ):
        super().__init__()

        self.token_num = token_num
        self.use_residual = use_residual
        self.mhc_iters = mhc_iters
        self.mhc_tau   = mhc_tau
        self.use_att = True

        # channel grouping
        self.group_channel_num = math.ceil(channels / token_num)
        self.channel_num = self.group_channel_num * token_num

        # ===== Local mHC generator =====
        self.local_mhc = LocalMHCGenerator(
            S=token_num,
            D=self.group_channel_num,
            hidden=128
        )

        # ===== Spatial branch (Mamba) =====
        self.spatial_branch = Mamba(
            d_model=self.group_channel_num,
            d_state=16,
            d_conv=4,
            expand=2,
        )

        self.spectral_branch = Mamba(
            d_model=1,
            d_state=16,
            d_conv=4,
            expand=2,
        )
        #self.spa_mamba = SpaMamba(self.group_channel_num, use_residual=use_residual, group_num=group_num)

        # ===== Projection =====
        self.proj = nn.Sequential(
            nn.GroupNorm(group_num, self.channel_num),
            nn.SiLU()
        )

        if self.use_att:
            self.weights = nn.Parameter(torch.ones(2) / 2)
            self.softmax = nn.Softmax(dim=0)

    def padding_feature(self, x):
        B, C, H, W = x.shape
        if C < self.channel_num:
            pad = torch.zeros(
                B, self.channel_num - C, H, W,
                device=x.device, dtype=x.dtype
            )
            x = torch.cat([x, pad], dim=1)
        return x

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        # 1) pad
        x_pad = self.padding_feature(x)   # (B,C',H,W)

        # 2) to tokens
        x_pad = x_pad.permute(0, 2, 3, 1).contiguous()
        B, H, W, C = x_pad.shape
        N = B * H * W

        R = x_pad.view(N, self.token_num, self.group_channel_num)  # (N,S,D)

        # 3) local mHC parameters
        H_pre_l, H_post_l, H_res_logits = self.local_mhc(R)

        # Eq.(8)
        H_pre  = torch.sigmoid(H_pre_l)            # (N,1,S)
        H_post = 2.0 * torch.sigmoid(H_post_l)     # (N,1,S)
        H_res  = sinkhorn_ds_batched(
            H_res_logits,
            iters=self.mhc_iters,
            tau=self.mhc_tau
        )                                           # (N,S,S)
        print(f"H_res first one Row {H_res[0].sum(dim=0)}")
        print(f"H_res first one Column {H_res[0].sum(dim=1)}")
        print(f"H_res first one Matrix {H_res[0]}")

        # 4) token mixing
        Rw = torch.einsum("nst,nsd->ntd", H_res, R)  # (N,S,D)


        # 5) aggregate → spatial
        x_aggr = torch.einsum("nks,nsd->nd", H_pre, R)  # (N,D)
        x_spatial = x_aggr.view(B, H, W, self.group_channel_num)
        x_spatial = x_spatial.permute(0, 3, 1, 2).contiguous()

        # 6) spatial branch
        xs = x_spatial.permute(0, 2, 3, 1).reshape(1, -1, self.group_channel_num)
        ys = self.spatial_branch(xs)
        x_pad = xs.reshape(B, H, W, self.group_channel_num)
        xs1 = x_pad.view(B * H * W, self.group_channel_num, 1)
        spe_ys = self.spectral_branch(xs1)
        spe_ys = spe_ys.reshape(B, H, W, self.group_channel_num).reshape(B, -1, self.group_channel_num)
        if self.use_att:
           weights = self.softmax(self.weights)
           fusion_x = ys * weights[0] + spe_ys * weights[1]
        #fusion = ys + spe_ys
        ys = fusion_x.view(B, H, W, self.group_channel_num)
        ys = ys.permute(0, 3, 1, 2).contiguous()

        # 7) distribute back to tokens
        y = ys.permute(0, 2, 3, 1).reshape(N, self.group_channel_num) 
        delta = torch.einsum("nd,nks->nsd", y, H_post)

        out = Rw + delta

        # 8) reconstruct
        x_rec = out.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_proj = self.proj(x_rec)

        if self.use_residual:
            x_proj = x_proj[:, :x.shape[1]]
            return x_proj
        else:
            x_proj = x_proj[:, :x.shape[1]]
            return x_proj



# ==================== UPDATED ARCHITECTURE ====================

class mHC_MambaHSI_Integrated(nn.Module):
    """
    Integrated architecture where:
    - Spectral tokens = mHC streams
    - H_res learns spectral correlations  
    - Spatial Mamba serves as branch function
    """
    
    def __init__(
        self,
        in_channels=128,
        hidden_dim=64,
        num_classes=10,
        token_num=8,           # Number of spectral tokens/mHC streams
        use_residual=True,
        group_num=4,
        use_dual_branch=False  # Option to keep original dual branch
    ):
        super().__init__()

        # Patch embedding (spectral reduction)
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, groups=token_num, out_channels=hidden_dim,
                     kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(token_num, hidden_dim),
            nn.SiLU()
        )
        
        # Multi-scale mHC-Spatial processing blocks
        self.blocks = nn.ModuleList([
                    SpectralStreamHyperConnections(
                        channels=hidden_dim,
                        token_num=token_num,
                        use_residual=use_residual,
                        group_num=token_num,
                    )
                    for i in range(3)
                ])
        # self.block1 = SpectralStreamHyperConnections(
        #     channels=hidden_dim,
        #     token_num=token_num,
        #     use_residual=use_residual,
        #     group_num=group_num,
        # )
        # self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        # self.block2 = SpectralStreamHyperConnections(
        #     channels=hidden_dim,
        #     token_num=token_num,
        #     use_residual=use_residual,
        #     group_num=group_num,
        # )
        # self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        # self.block3 = SpectralStreamHyperConnections(
        #     channels=hidden_dim,
        #     token_num=token_num,
        #     use_residual=use_residual,
        #     group_num=group_num,
        # )
        # self.blocks = nn.Sequential(
        #     # Block 1: Full resolution
        #     SpectralStreamHyperConnections(
        #         channels=hidden_dim,
        #         token_num=token_num,
        #         use_residual=use_residual,
        #         group_num=group_num,
        #         layer_index=0
        #     ),
        #     #nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            
        #     # Block 2: Half resolution
        #     SpectralStreamHyperConnections(
        #         channels=hidden_dim,
        #         token_num=token_num,
        #         use_residual=use_residual,
        #         group_num=group_num,
        #         layer_index=1
        #     ),
        #     #nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            
        #     # Block 3: Quarter resolution
        #     SpectralStreamHyperConnections(
        #         channels=hidden_dim,
        #         token_num=token_num,
        #         use_residual=use_residual,
        #         group_num=group_num,
        #         layer_index=2
        #     ),
        # )
        
        # Upsampling to recover spatial resolution
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=2, stride=2),
            nn.GroupNorm(group_num, hidden_dim),
            nn.SiLU(),
            
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=2, stride=2),
            nn.GroupNorm(group_num, hidden_dim),
            nn.SiLU(),
        )
        
        # Classification head
        self.cls_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 128, kernel_size=1, padding=0),
            nn.GroupNorm(group_num, 128),
            nn.SiLU(),
            
            nn.Conv2d(128, num_classes, kernel_size=1, stride=1, padding=0),
        )
    
    def forward(self, x):
        # 1. Spectral reduction
        x = self.patch_embedding(x)
        
        # 2. Multi-scale mHC-spatial processing
        for block in self.blocks:
            x = block(x)

        # x = self.block1(x)
        # x = self.pool1(x)
        # x = self.block2(x)
        # x = self.pool2(x)
        # x = self.block3(x)

        
        # 3. Upsample to original spatial size
        #x = self.upsample(x)
        
        # 4. Classification
        logits = self.cls_head(x)
        
        return logits


# ==================== DUAL-BRANCH VARIANT ====================

class DualBranch_mHC_MambaHSI(nn.Module):
    """
    Optional: Keep original dual-branch but replace SpeMamba with mHC version.
    This maintains both independent spatial and spectral pathways.
    """
    
    def __init__(self, in_channels=128, hidden_dim=64, num_classes=10, token_num=8):
        super().__init__()
        
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.GroupNorm(4, hidden_dim),
            nn.SiLU()
        )
        
        # Independent branches
        self.spectral_branch = SpectralStreamHyperConnections(
            channels=hidden_dim,
            token_num=token_num,
            use_residual=True
        )
        
        self.spatial_branch = nn.Sequential(
            SpaMamba(hidden_dim, use_residual=True),  # Original SpaMamba
            SpaMamba(hidden_dim, use_residual=True)
        )
        
        # Learnable fusion
        self.fusion_weights = nn.Parameter(torch.ones(2) / 2)
        self.softmax = nn.Softmax(dim=0)
        
        # Classification
        self.cls_head = nn.Sequential(
            nn.Conv2d(hidden_dim, num_classes, kernel_size=1),
        )
    
    def forward(self, x):
        x = self.patch_embedding(x)
        
        # Process independently
        x_spec = self.spectral_branch(x)
        x_spat = self.spatial_branch(x)
        
        # Adaptive fusion
        weights = self.softmax(self.fusion_weights)
        x_fused = x_spec * weights[0] + x_spat * weights[1]
        
        # Classification
        logits = self.cls_head(x_fused)
        
        return logits


# ==================== ANALYSIS & BENEFITS ====================

def analyze_integration():
    """
    Compare the integrated approach vs original.
    """
    print("=" * 60)
    print("ANALYSIS: mHC-Spectral Integration")
    print("=" * 60)
    
    # Original parameters
    original_params = sum(p.numel() for p in MambaHSI().parameters())
    
    # Integrated parameters  
    integrated_params = sum(p.numel() for p in mHC_MambaHSI_Integrated().parameters())
    
    print(f"\nParameter Comparison:")
    print(f"Original MambaHSI: {original_params:,}")
    print(f"Integrated mHC-MambaHSI: {integrated_params:,}")
    print(f"Difference: {integrated_params - original_params:,}")
    
    print(f"\nKey Integration Points:")
    print("1. Spectral tokens → mHC streams (natural alignment)")
    print("2. H_res matrix learns spectral correlations")
    print("3. H_pre aggregates spectral info for spatial processing")
    print("4. Spatial Mamba as branch function (preserves 2D structure)")
    print("5. H_post redistributes spatial features to spectral tokens")
    
    print(f"\nTheoretical Benefits:")
    print("✓ Explicit spectral correlation learning via H_res")
    print("✓ Doubly-stochastic constraints ensure stability")
    print("✓ Clear separation: spectral mixing → spatial processing")
    print("✓ Adaptive spectral aggregation via learnable H_pre/H_post")
    print("✓ Maintains spatial Mamba's efficiency for 2D processing")


# ==================== TEST ====================

if __name__ == "__main__":
    # Test the integrated architecture
    print("Testing Integrated mHC-MambaHSI Architecture...")
    
    # Create model
    model = mHC_MambaHSI_Integrated(
        in_channels=128,
        hidden_dim=64,
        num_classes=10,
        token_num=8
    )
    
    # Test input
    x = torch.randn(2, 128, 64, 64)  # HSI cube
    
    # Forward pass
    with torch.no_grad():
        logits = model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Run analysis
    analyze_integration()
    
    print("\n" + "=" * 60)
    print("✅ Integration Complete!")
    print("Architecture successfully replaces spectral Mamba with mHC")
    print("while using spatial Mamba as the branch function.")
    print("=" * 60)