import math
import pdb

import torch
from torch import nn
from mamba_ssm import Mamba
import torch.nn.functional as F

# ==================== SDMAMBA WITH K-MEANS CLUSTERING ====================
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


class ClusterHead(nn.Module):
    """K-means clustering head for semantic segmentation"""

    def __init__(self, feature_dim, num_clusters, temperature=1.0, ema_momentum=0.9):
        super().__init__()
        self.num_clusters = num_clusters
        self.temperature = temperature
        self.ema_momentum = ema_momentum
        self.cluster_centers = nn.Parameter(torch.ones(num_clusters, feature_dim))

    def update_centers(self, features, labels):
        """
        features: (1, C, H, W)
        labels:   (1, H, W), values in {-1,...,8}
        """
        B, C, H, W = features.shape
        features = features.permute(0, 2, 3, 1).reshape(-1, C)  # (N, C)
        labels = labels.view(-1)  # (N,)

        class_means = []
        for cls in range(self.num_clusters):
            mask = labels == cls
            if mask.any():
                mean_feat = features[mask].mean(dim=0)
            else:
                mean_feat = torch.zeros(C, device=features.device)
            class_means.append(mean_feat)
        class_means = torch.stack(class_means, dim=0)

        # EMA update
        with torch.no_grad():
            self.cluster_centers.data = (
                self.ema_momentum * self.cluster_centers.data
                + (1 - self.ema_momentum) * class_means
            )

    def forward(self, features):
        """
        Args:
            features: [B, L, D] where L = H * W (spatial positions), D = feature dim
        Returns:
            cluster_assignments: [B, L, K] soft cluster membership probabilities
        """
        B, L, D = features.shape

        # Normalize features and cluster centers
        features_norm = F.normalize(features, p=2, dim=-1)  # [B, L, D]
        centers_norm = F.normalize(self.cluster_centers, p=2, dim=-1)  # [K, D]

        # Compute similarity between features and cluster centers
        similarity = torch.matmul(features_norm, centers_norm.t())  # [B, L, K]

        # Convert to soft cluster assignments using temperature scaling
        cluster_assignments = F.softmax(similarity / self.temperature, dim=-1)

        return cluster_assignments

class KMeansSparseDeformableMambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, num_clusters=8, sparsity_ratio=1.0):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.expand = expand
        self.expanded_dim = dim * expand
        self.num_clusters = num_clusters
        self.sparsity_ratio = sparsity_ratio  # Now set to 1.0 to select all pixels

        self.norm = RMSNorm(dim)
        self.proj_in = nn.Linear(dim, self.expanded_dim)
        self.proj_out = nn.Linear(self.expanded_dim, dim)

        # K-means clustering head
        self.cluster_head = ClusterHead(self.expanded_dim, num_clusters)

        # SSM parameters
        self.A = nn.Parameter(torch.zeros(d_state, d_state))
        self.B = nn.Parameter(torch.zeros(1, 1, d_state))
        self.C = nn.Parameter(torch.zeros(self.expanded_dim, d_state))

        # Convolution layer
        self.conv = nn.Conv1d(
            in_channels=self.expanded_dim,
            out_channels=self.expanded_dim,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.expanded_dim,
            bias=False
        )


        self.mamba = Mamba(  # This module uses roughly 3 * expand * d_model^2 parameters
                           d_model=dim*2,  # Model dimension d_model
                           d_state=16,  # SSM state expansion factor
                           d_conv=4,  # Local convolution width
                           expand=2,  # Block expansion factor
                           )

    def compute_cluster_loss(self, features, cluster_assignments):
        """
        Compute clustering loss to maximize between-cluster variance and minimize within-cluster variance
        """
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
        # Compute pairwise distances between cluster centers
        centers_norm = F.normalize(cluster_centers, p=2, dim=-1)
        centers_similarity = torch.matmul(centers_norm, centers_norm.transpose(1, 2))  # [B, K, K]

        # Maximize distance between different clusters (minimize similarity)
        mask = 1 - torch.eye(K, device=features.device).unsqueeze(0)  # [1, K, K]
        between_cluster_sim = torch.sum(centers_similarity * mask) / (B * K * (K - 1))

        # Total clustering loss (minimize within-cluster, maximize between-cluster)
        cluster_loss = within_cluster_var + between_cluster_sim

        return cluster_loss

    def forward(self, x, return_cluster_assignments=False):
        B, L, C = x.shape
        residual = x

        # Normalize and project
        x_norm = self.norm(x)
        x_proj = self.proj_in(x_norm)  # [B, L, expanded_dim]

        # Get cluster assignments using K-means
        cluster_assignments = self.cluster_head(x_proj)  # [B, L, K]

        # With sparsity_ratio = 1.0, we select ALL pixels
        k_total = L  # This will be L when sparsity_ratio=1.0
        # if k_total >= L:
        #     # Process all pixels (no sparsity)
        #     selected_indices = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        #     x_sparse = x_proj
        # else:
            # Original sparse selection logic (kept for compatibility)
        k_per_cluster = max(1, int(L * 0.5 / self.num_clusters))
        selected_indices = []

        for cluster_idx in range(self.num_clusters):
            cluster_scores = cluster_assignments[:, :, cluster_idx]
            _, topk_indices = torch.topk(cluster_scores, k=k_per_cluster, dim=-1)
            selected_indices.append(topk_indices)

        selected_indices = torch.cat(selected_indices, dim=-1)
        if selected_indices.size(-1) > k_total:
            importance_scores = torch.gather(
                cluster_assignments.max(dim=-1)[0],
                1, selected_indices
            )
            _, top_importance_indices = torch.topk(importance_scores, k=k_total, dim=-1)
            selected_indices = torch.gather(selected_indices, 1, top_importance_indices)
        pdb.set_trace()
        x_sparse = batched_index_select(x_proj, 1, selected_indices)

        # Convolution processing

        x_conv = x_sparse.transpose(1, 2)
        x_conv = self.conv(x_conv)[..., :x_sparse.size(1)]
        x_conv = x_conv.transpose(1, 2)

        x_processed = self.mamba(x_conv)

        # SSM processing
        # h = torch.zeros(B, self.expanded_dim, self.d_state, device=x.device)
        # outputs = []
        #
        # for t in range(x_sparse.size(1)):
        #     x_t = x_conv[:, t].unsqueeze(-1)
        #     Bx = torch.sigmoid(self.B.to(x.device)) * x_t
        #     h = torch.matmul(h, self.A.to(x.device).T) + Bx
        #     out_t = (h * torch.sigmoid(self.C.to(x.device).unsqueeze(0))).sum(-1)
        #     outputs.append(out_t)
        #
        # x_processed = torch.stack(outputs, dim=1)
        x_processed = self.proj_out(x_processed)

        # Scatter back to original positions (when sparsity=1, this is identity)
        # if k_total >= L:
        #     output = x_processed
        # else:
        output = torch.zeros(B, L, C, device=x.device)
        output.scatter_(1, selected_indices.unsqueeze(-1).expand(-1, -1, C), x_processed)

        if return_cluster_assignments:
            cluster_loss = self.compute_cluster_loss(x_proj, cluster_assignments)

            return output + residual, cluster_assignments, cluster_loss
        else:
            return output + residual

class SpeMamba(nn.Module):
    def __init__(self,channels, token_num=8, use_residual=True, group_num=4):
        super(SpeMamba, self).__init__()
        self.token_num = token_num
        self.use_residual = use_residual

        self.group_channel_num = math.ceil(channels/token_num)
        self.channel_num = self.token_num * self.group_channel_num

        self.mamba = Mamba( # This module uses roughly 3 * expand * d_model^2 parameters
                            d_model=self.group_channel_num,  # Model dimension d_model
                            d_state=16,  # SSM state expansion factor
                            d_conv=4,  # Local convolution width
                            expand=2,  # Block expansion factor
                            )

        self.proj = nn.Sequential(
            nn.GroupNorm(group_num, self.channel_num),
            nn.SiLU()
        )

    def padding_feature(self,x):
        B, C, H, W = x.shape
        if C < self.channel_num:
            pad_c = self.channel_num - C
            pad_features = torch.zeros((B, pad_c, H, W)).to(x.device)
            cat_features = torch.cat([x, pad_features], dim=1)
            return cat_features
        else:
            return x

    def forward(self,x):
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

        # self.mamba = KMeansSparseDeformableMambaBlock(dim=channels, num_clusters=9, sparsity_ratio=1)
        if self.use_proj:
            self.proj = nn.Sequential(
                nn.GroupNorm(group_num, channels),
                nn.SiLU()
            )

    def forward(self,x):
        x_re = x.permute(0, 2, 3, 1).contiguous()
        B,H,W,C = x_re.shape
        x_flat = x_re.view(1,-1, C)
        #x_flat = self.mamba(x_flat, return_cluster_assignments=True)
        x_flat = self.mamba(x_flat)

        x_recon = x_flat.view(B, H, W, C)
        x_recon = x_recon.permute(0, 3, 1, 2).contiguous()
        if self.use_proj:
            x_recon = self.proj(x_recon)
        if self.use_residual:
            return x_recon + x
        else:
            return x_recon


class BothMamba(nn.Module):
    def __init__(self,channels,token_num,use_residual,group_num=4,use_att=True):
        super(BothMamba, self).__init__()
        self.use_att = use_att
        self.use_residual = use_residual
        if self.use_att:
            self.weights = nn.Parameter(torch.ones(2) / 2)
            self.softmax = nn.Softmax(dim=0)

        self.spa_mamba = SpaMamba(channels,use_residual=use_residual,group_num=group_num)
        self.spe_mamba = SpeMamba(channels,token_num=token_num,use_residual=use_residual,group_num=group_num)

    def forward(self,x):
        spa_x = self.spa_mamba(x)
        spe_x = self.spe_mamba(x)
        if self.use_att:
            weights = self.softmax(self.weights)
            fusion_x = spa_x * weights[0] + spe_x * weights[1]
        else:
            fusion_x = spa_x + spe_x
        if self.use_residual:
            return fusion_x + x
        else:
            return fusion_x


class MambaHSI(nn.Module):
    def __init__(self,in_channels=128,hidden_dim=64,num_classes=10,use_residual=True,mamba_type='both',token_num=4,group_num=4,use_att=True):
        super(MambaHSI, self).__init__()
        self.mamba_type = mamba_type

        self.patch_embedding = nn.Sequential(nn.Conv2d(in_channels=in_channels,out_channels=hidden_dim,kernel_size=1,stride=1,padding=0),
                                             nn.GroupNorm(group_num,hidden_dim),
                                             nn.SiLU())
        if mamba_type == 'spa':
            self.mamba = nn.Sequential(SpaMamba(hidden_dim,use_residual=use_residual,group_num=group_num),
                                        nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                                        SpaMamba(hidden_dim,use_residual=use_residual,group_num=group_num),
                                        nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                                        SpaMamba(hidden_dim,use_residual=use_residual,group_num=group_num),
                                        )
        elif mamba_type == 'spe':
            self.mamba = nn.Sequential(SpeMamba(hidden_dim,token_num=token_num,use_residual=use_residual,group_num=group_num),
                                        nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

                                        SpeMamba(hidden_dim,token_num=token_num,use_residual=use_residual,group_num=group_num),
                                        nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

                                        SpeMamba(hidden_dim,token_num=token_num,use_residual=use_residual,group_num=group_num)
                                        )

        elif mamba_type=='both':
            self.mamba1 = BothMamba(channels=hidden_dim,token_num=token_num,use_residual=use_residual,group_num=group_num,use_att=use_att)
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
            self.mamba2 = BothMamba(channels=hidden_dim, token_num=token_num, use_residual=use_residual, group_num=group_num,
                      use_att=use_att)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
            self.mamba3 = BothMamba(channels=hidden_dim,token_num=token_num,use_residual=use_residual,group_num=group_num,use_att=use_att)
            # self.mamba = nn.Sequential(BothMamba(channels=hidden_dim,token_num=token_num,use_residual=use_residual,group_num=group_num,use_att=use_att),
            #                            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            #
            #                            BothMamba(channels=hidden_dim,token_num=token_num,use_residual=use_residual,group_num=group_num,use_att=use_att),
            #                            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            #
            #                            BothMamba(channels=hidden_dim,token_num=token_num,use_residual=use_residual,group_num=group_num,use_att=use_att),
            #                            )


        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, 128),
            nn.GroupNorm(group_num, 128),
            nn.SiLU(),
            nn.Linear(128, num_classes),
        )

        self.proj_layer = nn.Conv2d(hidden_dim, hidden_dim*2, 1)
        self.cls_head = nn.Sequential(nn.Conv2d(in_channels=hidden_dim, out_channels=128, kernel_size=1, stride=1, padding=0),
                                      nn.GroupNorm(group_num,128),
                                      nn.SiLU(),
                                      nn.Conv2d(in_channels=128,out_channels=num_classes,kernel_size=1,stride=1,padding=0))

    def forward(self,x, labels=None):
        if self.training:
            x = self.patch_embedding(x)
            x = self.mamba1(x)
            #self.mamba1.spa_mamba.mamba.cluster_head.update_centers(self.proj_layer(x), labels)
            x = self.pool1(x)
            x = self.mamba2(x)
            x = self.pool2(x)
            x = self.mamba3(x)
            pdb.set_trace()
            logits = self.cls_head(x)
            return logits
        else:
            x = self.patch_embedding(x)
            x = self.mamba1(x)
            #x = self.pool1(x)
            x = self.mamba2(x)
            #x = self.pool2(x)
            x = self.mamba3(x)
            logits = self.cls_head(x)
            return logits



# if __name__=='__main__':
#     batch, length, dim = 2, 512*512, 256
#     x = torch.randn(batch, length, dim).to("cuda")
#     model = Mamba(
#         # This module uses roughly 3 * expand * d_model^2 parameters
#         d_model=dim,  # Model dimension d_model
#         d_state=16,  # SSM state expansion factor
#         d_conv=4,  # Local convolution width
#         expand=2,  # Block expansion factor
#     ).to("cuda")
#     y = model(x)
#     assert y.shape == x.shape