import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.filters as filters


class BGCDFLoss(nn.Module):
    """
    Vectorized Boundary-Guided Class Distribution Flow (BG-CDF) Loss
    """

    def __init__(
        self,
        num_classes: int = 2,
        profile_length: int = 5,
        boundary_threshold: float = 0.05,
        confidence_threshold: float = 0.7,
        max_boundaries: int = 1024,
        reduction: str = "mean",
    ):
        super().__init__()

        self.num_classes = num_classes
        self.profile_length = profile_length
        self.boundary_threshold = boundary_threshold
        self.confidence_threshold = confidence_threshold
        self.max_boundaries = max_boundaries
        self.reduction = reduction

    def forward(self, student_probs: torch.Tensor, teacher_probs: torch.Tensor):

        B, C, H, W = student_probs.shape
        device = student_probs.device
        eps = 1e-6

        # ------------------------------------
        # Teacher confidence mask
        # ------------------------------------
        teacher_conf, teacher_pred = torch.max(teacher_probs, dim=1)
        confidence_mask = teacher_conf > self.confidence_threshold

        # ------------------------------------
        # Boundary detection (Sobel)
        # ------------------------------------
        teacher_pred = teacher_pred.unsqueeze(1).float()
        grads = filters.spatial_gradient(teacher_pred, normalized=True)
        grad_x = grads[:, :, 0]
        grad_y = grads[:, :, 1]

        boundary_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2).squeeze(1)
        boundary_mask = boundary_mag > self.boundary_threshold

        # Combine confidence + boundary
        valid_mask = boundary_mask & confidence_mask

        total_loss = 0.0
        total_count = 0

        offsets = torch.linspace(
            -self.profile_length,
            self.profile_length,
            2 * self.profile_length + 1,
            device=device,
        )

        for b in range(B):

            ys, xs = torch.nonzero(valid_mask[b], as_tuple=True)

            if len(xs) == 0:
                continue

            # Subsample boundaries if too many
            if len(xs) > self.max_boundaries:
                idx = torch.randperm(len(xs), device=device)[: self.max_boundaries]
                xs = xs[idx]
                ys = ys[idx]

            dx = grad_x[b, 0, ys, xs]
            dy = grad_y[b, 0, ys, xs]

            norm = torch.sqrt(dx ** 2 + dy ** 2 + eps)
            dx = dx / norm
            dy = dy / norm

            # Vectorized profile sampling
            xs = xs.float().unsqueeze(1) + offsets.unsqueeze(0) * dx.unsqueeze(1)
            ys = ys.float().unsqueeze(1) + offsets.unsqueeze(0) * dy.unsqueeze(1)

            xs = torch.clamp(xs, 0, W - 1)
            ys = torch.clamp(ys, 0, H - 1)

            grid_x = 2 * xs / (W - 1) - 1
            grid_y = 2 * ys / (H - 1) - 1
            grid = torch.stack([grid_x, grid_y], dim=-1)
            grid = grid.unsqueeze(0)

            student_profile = F.grid_sample(
                student_probs[b].unsqueeze(0),
                grid,
                mode="bilinear",
                align_corners=True,
            )[0]

            teacher_profile = F.grid_sample(
                teacher_probs[b].unsqueeze(0),
                grid,
                mode="bilinear",
                align_corners=True,
            )[0]

            # CDF
            student_cdf = torch.cumsum(student_profile, dim=2)
            teacher_cdf = torch.cumsum(teacher_profile, dim=2)

            student_cdf = student_cdf / (student_cdf[:, :, -1:] + eps)
            teacher_cdf = teacher_cdf / (teacher_cdf[:, :, -1:] + eps)

            loss = F.mse_loss(student_cdf, teacher_cdf, reduction="sum")

            total_loss += loss
            total_count += student_cdf.numel()

        if total_count == 0:
            return torch.tensor(0.0, device=device)

        if self.reduction == "mean":
            return total_loss / total_count

        return total_loss



if __name__ == "__main__":
    # Quick test
    B, C, H, W = 2, 2, 64, 64
    student_probs = torch.rand(B, C, H, W)
    student_probs = student_probs / student_probs.sum(dim=1, keepdim=True)  # softmax
    teacher_probs = torch.rand(B, C, H, W)
    teacher_probs = teacher_probs / teacher_probs.sum(dim=1, keepdim=True)

    loss_fn = BGCDFLoss(num_classes=2, profile_length=5)
    loss_val = loss_fn(student_probs, teacher_probs)
    print("BG-CDF Loss:", loss_val.item())

