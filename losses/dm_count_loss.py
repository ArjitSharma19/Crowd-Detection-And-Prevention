import torch
import torch.nn as nn
from geomloss import SamplesLoss

class DMCountLoss(nn.Module):
    """
    Distribution Matching for Crowd Counting (DM-Count) Loss.
    Combines:
      1. Counting Loss: L1 distance of total count.
      2. Optimal Transport (OT) Loss: Sinkhorn distance between normalized distributions.
      3. Total Variation (TV) Loss: L1 distance between normalized predicted and discrete maps.
    Reference: Stony Brook cvlab-stonybrook/DM-Count
    """
    def __init__(self, lambda_ot=0.1, lambda_tv=0.01, ot_blur=0.05, device="cuda"):
        super(DMCountLoss, self).__init__()
        self.lambda_ot = lambda_ot
        self.lambda_tv = lambda_tv
        self.device = device
        # Initialize geomloss SamplesLoss with Sinkhorn backend and squared Euclidean cost
        self.ot_loss = SamplesLoss(loss="sinkhorn", p=2, blur=ot_blur, backend="tensorized")

    def forward(self, pred_density, gt_points, gt_counts):
        """
        Args:
            pred_density: predicted density maps (B, 1, H_target, W_target)
            gt_points: list of B tensors, each of shape (N_b, 2) of head coordinates in cropped space
            gt_counts: tensor of shape (B,) with total ground truth count
        """
        B, C, H_target, W_target = pred_density.shape
        device = pred_density.device

        # --- 1. Counting Loss ---
        # Sum of predicted density map represents predicted count
        pred_counts = pred_density.sum(dim=(1, 2, 3))
        count_loss = torch.mean(torch.abs(pred_counts - gt_counts))

        # --- 2. Optimal Transport (OT) Loss ---
        # Clamp predicted density to be non-negative
        pred_density_clamped = torch.clamp(pred_density, min=0.0)
        pred_sum = pred_density_clamped.sum(dim=(1, 2, 3), keepdim=True)
        
        # Robust normalization: if sum is zero, treat as uniform distribution to avoid NaN
        is_zero_pred = (pred_sum < 1e-6).float()
        pred_norm = (1.0 - is_zero_pred) * (pred_density_clamped / (pred_sum + 1e-8)) + is_zero_pred * (1.0 / (H_target * W_target))
        pred_weights = pred_norm.view(B, -1)  # shape: (B, H_target * W_target)

        # Generate grid coordinates in [0, 1] space
        y_coords, x_coords = torch.meshgrid(
            torch.linspace(0.0, 1.0, steps=H_target, device=device),
            torch.linspace(0.0, 1.0, steps=W_target, device=device),
            indexing='ij'
        )
        grid_coords = torch.stack([x_coords, y_coords], dim=-1).view(-1, 2)  # (H_target * W_target, 2)
        grid_coords = grid_coords.unsqueeze(0).repeat(B, 1, 1)  # (B, H_target * W_target, 2)

        total_ot_loss = 0.0
        active_ot_count = 0

        for b in range(B):
            g_coords = gt_points[b]
            n_pts = g_coords.shape[0]

            if n_pts == 0:
                # Skip OT loss for images with 0 people
                continue

            p_w = pred_weights[b]
            p_coords = grid_coords[b]
            
            g_w = torch.ones(n_pts, device=device) / n_pts

            # Normalize ground truth points to [0, 1] based on target grid scale
            # Raw cropped coordinates (x, y) map to target grid size (W_target*8, H_target*8)
            g_coords_norm = torch.zeros_like(g_coords)
            g_coords_norm[:, 0] = g_coords[:, 0] / (W_target * 8.0)
            g_coords_norm[:, 1] = g_coords[:, 1] / (H_target * 8.0)

            # Compute Sinkhorn divergence
            ot_loss_b = self.ot_loss(p_w, p_coords, g_w, g_coords_norm)
            total_ot_loss += ot_loss_b
            active_ot_count += 1

        ot_loss = total_ot_loss / max(1, active_ot_count) if active_ot_count > 0 else torch.tensor(0.0, device=device)

        # --- 3. Total Variation (TV) Loss ---
        # Generate discrete target map on target grid
        discrete_map = torch.zeros(B, 1, H_target, W_target, device=device)
        for b in range(B):
            pts = gt_points[b]
            if pts.shape[0] > 0:
                # Downsample coordinates by 8 to get target resolution index
                x_target = (pts[:, 0] / 8.0).long()
                y_target = (pts[:, 1] / 8.0).long()
                # Clamp boundaries to be safe
                x_target = torch.clamp(x_target, 0, W_target - 1)
                y_target = torch.clamp(y_target, 0, H_target - 1)
                for y, x in zip(y_target, x_target):
                    discrete_map[b, 0, y, x] += 1.0

        # Normalize discrete target map
        discrete_sum = discrete_map.sum(dim=(1, 2, 3), keepdim=True)
        is_zero_gt = (discrete_sum < 1e-6).float()
        discrete_norm = (1.0 - is_zero_gt) * (discrete_map / (discrete_sum + 1e-8)) + is_zero_gt * (1.0 / (H_target * W_target))

        # TV Loss: 0.5 * L1 distance between normalized maps
        tv_loss = 0.5 * torch.mean(torch.sum(torch.abs(pred_norm - discrete_norm), dim=(1, 2, 3)))

        # Combine
        loss = count_loss + self.lambda_ot * ot_loss + self.lambda_tv * tv_loss
        return loss, count_loss, ot_loss, tv_loss
