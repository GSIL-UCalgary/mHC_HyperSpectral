import math
import torch
from torch import nn
from mamba_ssm import Mamba
import torch.nn.functional as F
from model.SpectralAttention import SpeMambaWithAttention

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = x.norm(2, dim=-1, keepdim=True)
        rms_x = norm_x * (x.shape[-1] ** -0.5)
        return self.weight * (x / (rms_x + self.eps))


def batched_index_select(input, dim, index):
    """Select indices along specified dimension in batched manner"""
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)


# class AttentionSelector(nn.Module):
#     """Attention-based importance scoring for pixel selection"""
    
#     def __init__(self, feature_dim, num_heads=4):
#         super().__init__()
#         self.num_heads = num_heads
#         self.head_dim = feature_dim // num_heads
#         assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"
        
#         # Multi-head attention components
#         self.q_proj = nn.Linear(feature_dim, feature_dim)
#         self.k_proj = nn.Linear(feature_dim, feature_dim)
#         self.v_proj = nn.Linear(feature_dim, feature_dim)
#         self.out_proj = nn.Linear(feature_dim, 1)  # Project to importance score
        
#         self.scale = self.head_dim ** -0.5
        
#     def forward(self, x):
#         """
#         Args:
#             x: [B, L, D] features
#         Returns:
#             importance_scores: [B, L] attention-based importance scores
#         """
#         B, L, D = x.shape
        
#         # Multi-head projections
#         q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L, d]
#         k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L, d]
#         v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L, d]
        
#         # Compute attention scores
#         attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, L, L]
#         attn = F.softmax(attn, dim=-1)
        
#         # Apply attention to values
#         out = torch.matmul(attn, v)  # [B, H, L, d]
#         out = out.transpose(1, 2).contiguous().view(B, L, D)  # [B, L, D]
        
#         # Project to importance scores
#         importance_scores = self.out_proj(out).squeeze(-1)  # [B, L]
        
#         return importance_scores
class SimplifiedAttentionSelector(nn.Module):
    """Simplified attention without full multi-head - much lower memory"""
    
    def __init__(self, feature_dim, reduction_ratio=4):
        super().__init__()
        reduced_dim = feature_dim // reduction_ratio
        
        # Simple attention scoring
        self.query = nn.Linear(feature_dim, reduced_dim)
        self.key = nn.Linear(feature_dim, reduced_dim)
        self.scale = reduced_dim ** -0.5
        
    def forward(self, x):
        """
        Args:
            x: [B, L, D]
        Returns:
            importance_scores: [B, L]
        """
        B, L, D = x.shape
        
        q = self.query(x)  # [B, L, reduced_dim]
        k = self.key(x)    # [B, L, reduced_dim]
        
        # Compute self-attention scores (how much each position attends to itself)
        # This is much simpler than full attention
        importance_scores = torch.sum(q * k, dim=-1) * self.scale  # [B, L]
        
        return importance_scores

class ClusterHead(nn.Module):
    """Enhanced clustering with learnable temperature and better initialization"""
    
    def __init__(self, feature_dim, num_clusters, init_temperature=1.0, learn_temperature=True):
        super().__init__()
        self.num_clusters = num_clusters
        self.learn_temperature = learn_temperature
        
        if learn_temperature:
            self.temperature = nn.Parameter(torch.tensor(init_temperature))
        else:
            self.temperature = init_temperature
            
        # Better initialization using Xavier uniform
        self.cluster_centers = nn.Parameter(torch.Tensor(num_clusters, feature_dim))
        nn.init.xavier_uniform_(self.cluster_centers)
        
        # Cluster confidence weights
        self.cluster_confidence = nn.Parameter(torch.ones(num_clusters))
        
    def forward(self, features):
        B, L, D = features.shape
        
        # Compute distances instead of cosine similarity (often works better for clustering)
        features_expanded = features.unsqueeze(2)  # [B, L, 1, D]
        centers_expanded = self.cluster_centers.unsqueeze(0).unsqueeze(0)  # [1, 1, K, D]
        
        # Euclidean distance squared
        distances = torch.sum((features_expanded - centers_expanded) ** 2, dim=-1)  # [B, L, K]
        
        # Convert distances to similarities (closer = higher similarity)
        similarities = -distances
        
        # Apply temperature scaling
        if self.learn_temperature:
            temperature = torch.clamp(self.temperature, min=0.1, max=10.0)
        else:
            temperature = self.temperature
            
        cluster_assignments = F.softmax(similarities / temperature, dim=-1)
        
        # Apply cluster confidence weights
        confidence_weights = F.softmax(self.cluster_confidence, dim=0)
        weighted_assignments = cluster_assignments * confidence_weights.unsqueeze(0).unsqueeze(0)
        
        # Renormalize
        weighted_assignments = weighted_assignments / (weighted_assignments.sum(dim=-1, keepdim=True) + 1e-8)
        
        return weighted_assignments


class AttentionSparseDeformableMambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, num_clusters=19, 
                 sparsity_ratio=0.5, use_attention=True, num_heads=4, 
                 selection_mode='hybrid', use_cluster_weighting=True):
        """
        Args:
            use_cluster_weighting: Whether to use cluster information for feature weighting at sparsity=1.0
        """
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.expand = expand
        self.expanded_dim = dim * expand
        self.num_clusters = num_clusters
        self.sparsity_ratio = sparsity_ratio
        self.use_attention = use_attention
        self.selection_mode = selection_mode
        self.use_cluster_weighting = use_cluster_weighting

        self.norm = RMSNorm(dim)
        self.proj_in = nn.Linear(dim, self.expanded_dim)
        self.proj_out = nn.Linear(self.expanded_dim, dim)

        # Clustering for semantic segmentation
        self.cluster_head = ClusterHead(self.expanded_dim, num_clusters)
        
        # Attention-based importance scoring
        if use_attention:
            self.attention_selector = SimplifiedAttentionSelector(self.expanded_dim, num_heads)
            
        # Learnable weights for combining attention and clustering
        if selection_mode == 'hybrid':
            self.selection_weight = nn.Parameter(torch.tensor(0.5))
            
        # Learnable per-cluster importance weights (for sparsity=1.0)
        if use_cluster_weighting and selection_mode in ['cluster', 'hybrid']:
            self.cluster_importance = nn.Parameter(torch.ones(num_clusters) / num_clusters)

        # Convolution layer
        self.conv = nn.Conv1d(
            in_channels=self.expanded_dim,
            out_channels=self.expanded_dim,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.expanded_dim,
            bias=False
        )

        # Mamba SSM
        self.mamba = Mamba(
            d_model=dim*2,
            d_state=16,
            d_conv=4,
            expand=2,
        )

    def compute_cluster_weights(self, cluster_assignments):
        """
        Compute per-pixel weights based on cluster membership
        
        Args:
            cluster_assignments: [B, L, K] soft cluster probabilities
        Returns:
            cluster_weights: [B, L, 1] aggregated cluster importance weights
        """
        # Normalize cluster importance scores
        importance_scores = F.softmax(self.cluster_importance, dim=0)  # [K]
        
        # Weighted sum: each pixel gets weight based on which clusters it belongs to
        cluster_weights = torch.matmul(cluster_assignments, importance_scores)  # [B, L]
        
        return cluster_weights.unsqueeze(-1)  # [B, L, 1]

    def apply_feature_weighting(self, x_proj, cluster_assignments, attention_scores):
        """
        Apply feature weighting using cluster and/or attention information
        
        Args:
            x_proj: [B, L, C] projected features
            cluster_assignments: [B, L, K] cluster probabilities
            attention_scores: [B, L] attention importance scores (or None)
        Returns:
            weighted_features: [B, L, C]
        """
        B, L, C = x_proj.shape
        
        if self.selection_mode == 'attention' and attention_scores is not None:
            # Pure attention weighting
            attention_weights = F.softmax(attention_scores, dim=-1).unsqueeze(-1)  # [B, L, 1]
            return x_proj * attention_weights
            
        elif self.selection_mode == 'cluster' and self.use_cluster_weighting:
            # Pure cluster weighting
            cluster_weights = self.compute_cluster_weights(cluster_assignments)  # [B, L, 1]
            return x_proj * cluster_weights
            
        elif self.selection_mode == 'hybrid':
            # Combine both attention and cluster information
            weights_list = []
            
            if attention_scores is not None:
                attention_weights = F.softmax(attention_scores, dim=-1).unsqueeze(-1)  # [B, L, 1]
                weights_list.append(attention_weights)
            
            if self.use_cluster_weighting:
                cluster_weights = self.compute_cluster_weights(cluster_assignments)  # [B, L, 1]
                weights_list.append(cluster_weights)
            
            if len(weights_list) == 2:
                # Learn optimal combination
                alpha = torch.sigmoid(self.selection_weight)
                combined_weights = alpha * weights_list[0] + (1 - alpha) * weights_list[1]
            elif len(weights_list) == 1:
                combined_weights = weights_list[0]
            else:
                combined_weights = 1.0
                
            return x_proj * combined_weights
        else:
            # No weighting
            return x_proj

    def select_pixels(self, x_proj, cluster_assignments, k_total, attention_scores=None):
        """Select top-k pixels based on selection mode"""
        B, L, C = x_proj.shape
        
        if self.selection_mode == 'cluster':
            # Original cluster-based selection
            k_per_cluster = max(1, int(k_total / self.num_clusters))
            selected_indices = []

            for cluster_idx in range(self.num_clusters):
                cluster_scores = cluster_assignments[:, :, cluster_idx]
                _, topk_indices = torch.topk(cluster_scores, k=k_per_cluster, dim=-1)
                selected_indices.append(topk_indices)

            selected_indices = torch.cat(selected_indices, dim=-1)
            
            # Trim if necessary
            if selected_indices.size(-1) > k_total:
                importance_scores = torch.gather(
                    cluster_assignments.max(dim=-1)[0],
                    1, selected_indices
                )
                _, top_importance_indices = torch.topk(importance_scores, k=k_total, dim=-1)
                selected_indices = torch.gather(selected_indices, 1, top_importance_indices)
                
        elif self.selection_mode == 'attention' and attention_scores is not None:
            # Pure attention-based selection
            _, selected_indices = torch.topk(attention_scores, k=k_total, dim=-1)
            
        elif self.selection_mode == 'hybrid' and attention_scores is not None:
            # Hybrid: combine clustering diversity with attention importance
            attention_scores_norm = F.softmax(attention_scores, dim=-1)
            cluster_scores = cluster_assignments.max(dim=-1)[0]  # [B, L]
            cluster_scores_norm = F.softmax(cluster_scores, dim=-1)
            
            # Combine with learned weight
            alpha = torch.sigmoid(self.selection_weight)
            combined_scores = alpha * attention_scores_norm + (1 - alpha) * cluster_scores_norm
            
            _, selected_indices = torch.topk(combined_scores, k=k_total, dim=-1)
        else:
            # Fallback: random or first k
            selected_indices = torch.arange(k_total, device=x_proj.device).unsqueeze(0).expand(B, -1)
        
        return selected_indices

    def forward(self, x, return_cluster_assignments=False):
        B, L, C = x.shape
        residual = x

        # Normalize and project
        x_norm = self.norm(x)
        x_proj = self.proj_in(x_norm)  # [B, L, expanded_dim]

        # Get cluster assignments
        cluster_assignments = self.cluster_head(x_proj)  # [B, L, K]

        # Determine number of pixels to select
        k_total = max(1, int(L * self.sparsity_ratio))
        
        # Compute attention scores if needed
        attention_scores = None
        if self.use_attention and self.selection_mode in ['attention', 'hybrid']:
            attention_scores = self.attention_selector(x_proj)  # [B, L]
       
        if k_total >= L:
            # Process all pixels - USE BOTH cluster and attention information
            selected_indices = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
            
            # Apply feature weighting based on cluster and/or attention
            x_sparse = self.apply_feature_weighting(x_proj, cluster_assignments, attention_scores)
        else:
            # Select pixels based on selection mode
            selected_indices = self.select_pixels(x_proj, cluster_assignments, k_total, attention_scores)
            x_sparse = batched_index_select(x_proj, 1, selected_indices)

        # Convolution processing
        x_conv = x_sparse.transpose(1, 2)
        x_conv = self.conv(x_conv)[..., :x_sparse.size(1)]
        x_conv = x_conv.transpose(1, 2)

        # Mamba processing
        x_processed = self.mamba(x_conv)
        x_processed = self.proj_out(x_processed)

        # Scatter back to original positions
        # ALWAYS SCATTER BACK - maintaining spatial consistency
        output = torch.zeros(B, L, C, device=x.device)
        output.scatter_(1, selected_indices.unsqueeze(-1).expand(-1, -1, C), x_processed)

        if return_cluster_assignments:
            cluster_loss = self.compute_cluster_loss(x_proj, cluster_assignments)
            return output + residual, cluster_assignments, cluster_loss
        else:
            return output + residual

    def compute_cluster_loss(self, features, cluster_assignments):
        """Compute clustering loss for semantic segmentation"""
        B, L, D = features.shape
        K = self.num_clusters

        # Normalize features
        features_norm = F.normalize(features, p=2, dim=-1)

        # Compute cluster centers from assignments
        cluster_weights = cluster_assignments.sum(dim=1)  # [B, K]
        weighted_features = torch.matmul(cluster_assignments.transpose(1, 2), features_norm)  # [B, K, D]

        # Avoid division by zero
        cluster_weights = cluster_weights + 1e-8
        cluster_centers = weighted_features / cluster_weights.unsqueeze(-1)  # [B, K, D]

        # Within-cluster variance
        expanded_centers = cluster_centers.unsqueeze(1)  # [B, 1, K, D]
        expanded_features = features_norm.unsqueeze(2)  # [B, L, 1, D]

        # Compute distances to cluster centers
        distances = torch.sum((expanded_features - expanded_centers) ** 2, dim=-1)  # [B, L, K]

        # Weighted within-cluster variance
        within_cluster_var = torch.sum(cluster_assignments * distances) / (B * L)

        # Between-cluster variance
        centers_norm = F.normalize(cluster_centers, p=2, dim=-1)
        centers_similarity = torch.matmul(centers_norm, centers_norm.transpose(1, 2))  # [B, K, K]

        # Maximize distance between different clusters
        mask = 1 - torch.eye(K, device=features.device).unsqueeze(0)  # [1, K, K]
        between_cluster_sim = torch.sum(centers_similarity * mask) / (B * K * (K - 1))

        # Total clustering loss
        cluster_loss = within_cluster_var + between_cluster_sim

        return cluster_loss


class SpeMamba(nn.Module):
    def __init__(self, channels, token_num=8, use_residual=True, group_num=4):
        super(SpeMamba, self).__init__()
        self.token_num = token_num
        self.use_residual = use_residual

        self.group_channel_num = math.ceil(channels/token_num)
        self.channel_num = self.token_num * self.group_channel_num

        self.mamba = Mamba(
            d_model=self.group_channel_num,
            d_state=16,
            d_conv=4,
            expand=2,
        )

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
        x_pad = self.padding_feature(x)
        x_pad = x_pad.permute(0, 2, 3, 1).contiguous()
        B, H, W, C_pad = x_pad.shape
        x_flat = x_pad.view(B * H * W, self.token_num, self.group_channel_num)
        x_flat = self.mamba(x_flat)
        x_recon = x_flat.view(B, H, W, C_pad)
        x_recon = x_recon.permute(0, 3, 1, 2).contiguous()
        x_proj = self.proj(x_recon)
        if self.use_residual:
            return x + x_proj
        else:
            return x_proj


class SpaMamba(nn.Module):
    def __init__(self, channels, use_residual=True, group_num=4, use_proj=True,
                 num_clusters=9, sparsity_ratio=1.0, use_attention=True, 
                 num_heads=4, selection_mode='hybrid'):
        super(SpaMamba, self).__init__()
        self.use_residual = use_residual
        self.use_proj = use_proj

        self.mamba = AttentionSparseDeformableMambaBlock(
            dim=channels, 
            num_clusters=num_clusters, 
            sparsity_ratio=sparsity_ratio,
            use_attention=use_attention,
            num_heads=num_heads,
            selection_mode=selection_mode,
            use_cluster_weighting=True  # NEW: Enable cluster weighting
        )
        
        if self.use_proj:
            self.proj = nn.Sequential(
                nn.GroupNorm(group_num, channels),
                nn.SiLU()
            )

    def forward(self, x):
        x_re = x.permute(0, 2, 3, 1).contiguous()
        B, H, W, C = x_re.shape
        x_flat = x_re.view(1, -1, C)
        x_flat, cluster_assignments, cluster_loss = self.mamba(x_flat, return_cluster_assignments=True)

        x_recon = x_flat.view(B, H, W, C)
        x_recon = x_recon.permute(0, 3, 1, 2).contiguous()
        if self.use_proj:
            x_recon = self.proj(x_recon)
        if self.use_residual:
            return x_recon + x, cluster_assignments, cluster_loss
        else:
            return x_recon


class BothMamba(nn.Module):
    def __init__(self, channels, token_num, use_residual, group_num=4, use_att=True,
                 num_clusters=9, sparsity_ratio=1.0, attention_heads=4, 
                 selection_mode='hybrid'):
        super(BothMamba, self).__init__()
        self.use_att = use_att
        self.use_residual = use_residual
        if self.use_att:
            self.weights = nn.Parameter(torch.ones(2) / 2)
            self.softmax = nn.Softmax(dim=0)

        self.spa_mamba = SpaMamba(
            channels, 
            use_residual=use_residual, 
            group_num=group_num,
            num_clusters=num_clusters,
            sparsity_ratio=sparsity_ratio,
            use_attention=True,
            num_heads=attention_heads,
            selection_mode=selection_mode
        )
        self.spe_mamba = SpeMamba(channels, token_num=token_num, use_residual=use_residual, group_num=group_num)

    def forward(self, x):
        spa_x, cluster_assignments, cluster_loss = self.spa_mamba(x)
        spe_x = self.spe_mamba(x)
        if self.use_att:
            weights = self.softmax(self.weights)
            fusion_x = spa_x * weights[0] + spe_x * weights[1]
        else:
            fusion_x = spa_x + spe_x
        if self.use_residual:
            return fusion_x + x, cluster_assignments, cluster_loss
        else:
            return fusion_x


class MambaHSI(nn.Module):
    def __init__(self, in_channels=128, hidden_dim=64, num_classes=10, 
                 use_residual=True, mamba_type='both', token_num=4, group_num=4, 
                 use_att=True, num_clusters=9, sparsity_ratio=1.0, 
                 attention_heads=4, selection_mode='hybrid'):
        """
        Args:
            selection_mode: 'attention', 'cluster', or 'hybrid' for pixel selection strategy
            attention_heads: number of attention heads for importance scoring
            sparsity_ratio: ratio of pixels to select (1.0 = all pixels)
        """
        super(MambaHSI, self).__init__()
        self.mamba_type = mamba_type

        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(group_num, hidden_dim),
            nn.SiLU()
        )
        
        if mamba_type == 'both':
            self.mamba1 = BothMamba(
                channels=hidden_dim, 
                token_num=token_num, 
                use_residual=use_residual, 
                group_num=group_num, 
                use_att=use_att,
                num_clusters=num_clusters,
                sparsity_ratio=sparsity_ratio,
                attention_heads=attention_heads,
                selection_mode=selection_mode
            )
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
            
            self.mamba2 = BothMamba(
                channels=hidden_dim, 
                token_num=token_num, 
                use_residual=use_residual, 
                group_num=group_num,
                use_att=use_att,
                num_clusters=num_clusters,
                sparsity_ratio=sparsity_ratio,
                attention_heads=attention_heads,
                selection_mode=selection_mode
            )
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
            
            self.mamba3 = BothMamba(
                channels=hidden_dim, 
                token_num=token_num, 
                use_residual=use_residual, 
                group_num=group_num, 
                use_att=use_att,
                num_clusters=num_clusters,
                sparsity_ratio=sparsity_ratio,
                attention_heads=attention_heads,
                selection_mode=selection_mode
            )

        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(group_num, 128),
            nn.SiLU(),
            nn.Conv2d(in_channels=128, out_channels=num_classes, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        x, _, cluster_loss0 = self.mamba1(x)
        x = self.pool1(x)
        x, _, cluster_loss1 = self.mamba2(x)
        x = self.pool2(x)
        x, _, cluster_loss2 = self.mamba3(x)

        logits = self.cls_head(x)
        return logits, cluster_loss2 + cluster_loss0 + cluster_loss1
        
        
#  '''
#  To Do:
#  Change Transformer attention from linear layer
#  update SPE module to have attention or something
#  Local mamba + global in Spa
#  Sigmoid may not make sense for alpha
#  Check clusterng score to make sure it grabs index
#  torch.topk may not be right
 
#  '''

# Baseline alpha = 0
#   ====================Starting evaluation for testing set.========================
# [I 251007 15:03:06 train_MambaHSI:166] Test 199|OA:0.956738023387401|MACC:0.9593293973807729|Kappa:0.9500000000000002|MIOU:0.890752084229737|IOU:[0.87655952 0.96615072 0.8474359  0.75075487 0.97752809 0.98393176
#      0.93192133 0.8678392  0.81464738]|ACC:[0.90608405 0.96936966 0.96308888 0.90443122 1.         0.99418721
#      0.95503876 0.94838001 0.99338479]
# [I 251007 15:03:06 train_MambaHSI:166] 
#     ====================== exp_idx=0 seed=0 learning rate=0.0003 epochs=200 train ratio=0.1 val ratio=0.01 ======================
#     OA=0.956738023387401
#     AA=0.9593293973807729
#     kpp=0.9500000000000002
#     mIOU_test:0.890752084229737
#     IOU_test:[0.87655952 0.96615072 0.8474359  0.75075487 0.97752809 0.98393176
#      0.93192133 0.8678392  0.81464738]
#     Acc_test:[0.90608405 0.96936966 0.96308888 0.90443122 1.         0.99418721
#      0.95503876 0.94838001 0.99338479]

# With SPE Attention 
#     ====================Starting evaluation for testing set.========================
    
# [I 251007 15:24:03 train_MambaHSI:166] Test 199|OA:0.9486043002640513|MACC:0.9570510482143268|Kappa:0.9750000000000001|MIOU:0.8723547903428382|IOU:[0.82834186 0.96500911 0.78416149 0.74723197 0.9871407  0.98631495
#      0.79724656 0.86444213 0.89130435]|ACC:[0.860264   0.96775754 0.98105877 0.91501323 1.         0.99679294
#      0.9875969  0.91048874 0.99448732]
# [I 251007 15:24:03 train_MambaHSI:166] 
#     ====================== exp_idx=0 seed=0 learning rate=0.0003 epochs=200 train ratio=0.1 val ratio=0.01 ======================
#     OA=0.9486043002640513
#     AA=0.9570510482143268
#     kpp=0.9750000000000001
#     mIOU_test:0.8723547903428382
#     IOU_test:[0.82834186 0.96500911 0.78416149 0.74723197 0.9871407  0.98631495
#      0.79724656 0.86444213 0.89130435]
#     Acc_test:[0.860264   0.96775754 0.98105877 0.91501323 1.         0.99679294
#      0.9875969  0.91048874 0.99448732]

# With Sigmoid instead of alpha =0
#     ====================Starting evaluation for testing set.========================
    
# [I 251007 15:59:22 train_MambaHSI:166] Test 199|OA:0.9638815541305168|MACC:0.9577529559476594|Kappa:0.9500000000000002|MIOU:0.9027457127627285|IOU:[0.89264243 0.97825272 0.89442139 0.79564586 0.98639456 0.98416469
#      0.90685544 0.85253931 0.83379501]|ACC:[0.94613867 0.98140685 0.94220495 0.90641534 1.         0.9965925
#      0.94341085 0.90801757 0.99558986]
# [I 251007 15:59:22 train_MambaHSI:166] 
#     ====================== exp_idx=0 seed=0 learning rate=0.0003 epochs=200 train ratio=0.1 val ratio=0.01 ======================
#     OA=0.9638815541305168
#     AA=0.9577529559476594
#     kpp=0.9500000000000002
#     mIOU_test:0.9027457127627285
#     IOU_test:[0.89264243 0.97825272 0.89442139 0.79564586 0.98639456 0.98416469
#      0.90685544 0.85253931 0.83379501]
#     Acc_test:[0.94613867 0.98140685 0.94220495 0.90641534 1.         0.9965925
#      0.94341085 0.90801757 0.99558986]

# Clusters are assignments instead of values 
#     ====================Starting evaluation for testing set.========================
    
# [I 251007 16:10:00 train_MambaHSI:166] Test 199|OA:0.9462702753677857|MACC:0.9608292661698923|Kappa:0.9500000000000002|MIOU:0.876274195537844|IOU:[0.86781693 0.94392523 0.90178571 0.67701122 0.97899475 0.98550149
#      0.82084691 0.87307391 0.83751161]|ACC:[0.90047034 0.94438175 0.98105877 0.93783069 1.         0.99458809
#      0.97674419 0.91790225 0.99448732]
# [I 251007 16:10:00 train_MambaHSI:166] 
#     ====================== exp_idx=0 seed=0 learning rate=0.0003 epochs=200 train ratio=0.1 val ratio=0.01 ======================
#     OA=0.9462702753677857
#     AA=0.9608292661698923
#     kpp=0.9500000000000002
#     mIOU_test:0.876274195537844
#     IOU_test:[0.86781693 0.94392523 0.90178571 0.67701122 0.97899475 0.98550149
#      0.82084691 0.87307391 0.83751161]
#     Acc_test:[0.90047034 0.94438175 0.98105877 0.93783069 1.         0.99458809
#      0.97674419 0.91790225 0.99448732]