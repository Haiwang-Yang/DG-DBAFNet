from pathlib import Path
from typing import Tuple
from urllib.parse import urlparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


TRAIN_MODES = ["linear_probe", "full_finetune", "no_pretrain"]
BACKBONE_INITS = ["imagenet", "random"]
BRANCH_MODES = ["dual", "acoustic_only", "backbone_only", "backbone_only_matched"]
FUSION_TYPES = ["agf", "concat", "none"]


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.alpha * grad_output, None


def grad_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return GradientReversalFunction.apply(x, alpha)


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        weights = self.pool(x).view(b, c)
        weights = self.mlp(weights).view(b, c, 1, 1)
        return x * weights


class SpatialAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.ReLU(inplace=True),
        )
        self.global_conv = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.global_conv(self.local(x))
        return x * weights


class MixStyle(nn.Module):
    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or torch.rand(1, device=x.device).item() > self.p or x.size(0) < 2:
            return x
        b = x.size(0)
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True, unbiased=False)
        sig = (var + self.eps).sqrt()
        normalized = (x - mu) / sig
        perm = torch.randperm(b, device=x.device)
        lmda = torch.distributions.Beta(self.alpha, self.alpha).sample((b, 1, 1, 1)).to(x.device)
        mu_mix = mu * lmda + mu[perm] * (1 - lmda)
        sig_mix = sig * lmda + sig[perm] * (1 - lmda)
        return normalized * sig_mix + mu_mix


class AcousticAttentionBranch(nn.Module):
    def __init__(
        self,
        out_dim: int = 256,
        use_channel_attention: bool = True,
        use_spatial_attention: bool = True,
        mixstyle_p: float = 0.5,
        mixstyle_alpha: float = 0.1,
    ):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.mixstyle = MixStyle(p=mixstyle_p, alpha=mixstyle_alpha) if mixstyle_p > 0 else nn.Identity()
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.channel_attention = ChannelAttention(128) if use_channel_attention else nn.Identity()
        self.spatial_attention = SpatialAttention(128) if use_spatial_attention else nn.Identity()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.mixstyle(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return self.head(x)


def _replace_classifier_with_identity(model: nn.Module) -> Tuple[nn.Module, int]:
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        out_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, out_dim
    if hasattr(model, "classifier"):
        classifier = model.classifier
        if isinstance(classifier, nn.Linear):
            out_dim = classifier.in_features
            model.classifier = nn.Identity()
            return model, out_dim
        if isinstance(classifier, nn.Sequential):
            for i in range(len(classifier) - 1, -1, -1):
                layer = classifier[i]
                if isinstance(layer, nn.Linear):
                    out_dim = layer.in_features
                    classifier[i] = nn.Identity()
                    return model, out_dim
                if isinstance(layer, nn.Conv2d):
                    out_dim = layer.in_channels
                    classifier[i] = nn.Identity()
                    return model, out_dim
    if hasattr(model, "head") and isinstance(model.head, nn.Linear):
        out_dim = model.head.in_features
        model.head = nn.Identity()
        return model, out_dim
    if hasattr(model, "heads") and hasattr(model.heads, "head"):
        out_dim = model.heads.head.in_features
        model.heads.head = nn.Identity()
        return model, out_dim
    raise ValueError(f"Could not replace classifier for model {type(model).__name__}")


class PretrainedBackboneBranch(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet18",
        train_mode: int = 1,
        out_dim: int = 256,
        dropout: float = 0.3,
        backbone_init: str = "imagenet",
        allow_weight_download: bool = False,
    ):
        super().__init__()
        if train_mode not in range(len(TRAIN_MODES)):
            raise ValueError(f"Unsupported train mode: {train_mode}")
        if backbone_init not in BACKBONE_INITS:
            raise ValueError(f"Unsupported backbone initialization: {backbone_init}")
        if not hasattr(models, backbone):
            raise ValueError(f"Unsupported torchvision model: {backbone}")

        weights = None
        if backbone_init == "imagenet" and hasattr(models, "get_model_weights"):
            try:
                candidate = models.get_model_weights(backbone).DEFAULT
                checkpoint_name = Path(urlparse(candidate.url).path).name
                checkpoint_path = Path(torch.hub.get_dir()) / "checkpoints" / checkpoint_name
                if checkpoint_path.exists() or allow_weight_download:
                    weights = candidate
                else:
                    raise FileNotFoundError(
                        f"ImageNet weights are not cached at {checkpoint_path}. "
                        "Download them explicitly or pass allow_weight_download=True."
                    )
            except FileNotFoundError:
                raise
            except Exception:
                weights = None
        self.backbone_init = backbone_init
        self.model = getattr(models, backbone)(weights=weights)
        self.model, self.raw_out_dim = _replace_classifier_with_identity(self.model)
        for parameter in self.model.parameters():
            parameter.requires_grad = train_mode != 0
        self.projection = nn.Sequential(
            nn.Linear(self.raw_out_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.model(x)
        if y.ndim > 2:
            y = torch.flatten(y, 1)
        return self.projection(y)


class AdaptiveGatedFusion(nn.Module):
    def __init__(self, branch_dim: int = 256, fused_dim: int = 512):
        super().__init__()
        self.acoustic_projection = nn.Linear(branch_dim, fused_dim)
        self.backbone_projection = nn.Linear(branch_dim, fused_dim)
        self.gate = nn.Sequential(
            nn.Linear(branch_dim * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, fused_dim),
            nn.Sigmoid(),
        )

    def forward(self, acoustic_features: torch.Tensor, backbone_features: torch.Tensor) -> torch.Tensor:
        acoustic_projected = self.acoustic_projection(acoustic_features)
        backbone_projected = self.backbone_projection(backbone_features)
        gate = self.gate(torch.cat([acoustic_features, backbone_features], dim=1))
        return gate * acoustic_projected + (1.0 - gate) * backbone_projected


class ResidualCapacityAdapter(nn.Module):
    """Add trainable capacity to a single branch without adding acoustic input cues."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive.")
        self.hidden_dim = hidden_dim
        self.layers = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)


def _module_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def capacity_matching_hidden_dim(
    branch_dim: int,
    fusion_dim: int,
    use_channel_attention: bool,
    use_spatial_attention: bool,
    mixstyle_p: float,
    mixstyle_alpha: float,
) -> int:
    acoustic = AcousticAttentionBranch(
        out_dim=branch_dim,
        use_channel_attention=use_channel_attention,
        use_spatial_attention=use_spatial_attention,
        mixstyle_p=mixstyle_p,
        mixstyle_alpha=mixstyle_alpha,
    )
    full_fusion = AdaptiveGatedFusion(branch_dim=branch_dim, fused_dim=fusion_dim)
    single_fusion = nn.Linear(branch_dim, fusion_dim)
    target_extra = (
        _module_parameters(acoustic)
        + _module_parameters(full_fusion)
        - _module_parameters(single_fusion)
    )
    # A dim -> hidden -> dim MLP has (2 * dim + 1) * hidden + dim parameters.
    return max(1, round((target_extra - branch_dim) / (2 * branch_dim + 1)))


class AblationDGDBAFNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 4,
        backbone: str = "resnet18",
        train_mode: int = 1,
        branch_mode: str = "dual",
        fusion_type: str = "agf",
        branch_dim: int = 256,
        fusion_dim: int = 512,
        projection_dim: int = 128,
        num_domains: int = 4,
        grl_alpha: float = 1.0,
        use_channel_attention: bool = True,
        use_spatial_attention: bool = True,
        mixstyle_p: float = 0.5,
        mixstyle_alpha: float = 0.1,
        backbone_init: str = "imagenet",
        capacity_match_hidden: int = 0,
        allow_weight_download: bool = False,
    ):
        super().__init__()
        if branch_mode not in BRANCH_MODES:
            raise ValueError(f"Unsupported branch_mode: {branch_mode}")
        if fusion_type not in FUSION_TYPES:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")
        if branch_mode != "dual" and fusion_type != "none":
            raise ValueError("Single-branch ablations must use fusion_type='none'.")
        if branch_mode == "dual" and fusion_type == "none":
            raise ValueError("Dual-branch ablations must use fusion_type='agf' or 'concat'.")

        self.branch_mode = branch_mode
        self.fusion_type = fusion_type
        self.grl_alpha = grl_alpha
        self.backbone_init = backbone_init
        self.acoustic_branch = None
        self.backbone_branch = None
        self.capacity_adapter = nn.Identity()

        if branch_mode in {"dual", "acoustic_only"}:
            self.acoustic_branch = AcousticAttentionBranch(
                out_dim=branch_dim,
                use_channel_attention=use_channel_attention,
                use_spatial_attention=use_spatial_attention,
                mixstyle_p=mixstyle_p,
                mixstyle_alpha=mixstyle_alpha,
            )
        if branch_mode in {"dual", "backbone_only", "backbone_only_matched"}:
            self.backbone_branch = PretrainedBackboneBranch(
                backbone=backbone,
                train_mode=train_mode,
                out_dim=branch_dim,
                backbone_init=backbone_init,
                allow_weight_download=allow_weight_download,
            )

        if branch_mode == "backbone_only_matched":
            hidden_dim = capacity_match_hidden or capacity_matching_hidden_dim(
                branch_dim=branch_dim,
                fusion_dim=fusion_dim,
                use_channel_attention=use_channel_attention,
                use_spatial_attention=use_spatial_attention,
                mixstyle_p=mixstyle_p,
                mixstyle_alpha=mixstyle_alpha,
            )
            self.capacity_adapter = ResidualCapacityAdapter(branch_dim, hidden_dim)

        if branch_mode == "dual" and fusion_type == "agf":
            self.fusion = AdaptiveGatedFusion(branch_dim=branch_dim, fused_dim=fusion_dim)
        elif branch_mode == "dual" and fusion_type == "concat":
            self.fusion = nn.Identity()
        else:
            self.fusion = nn.Linear(branch_dim, fusion_dim)

        self.fusion_norm = nn.Sequential(
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
        self.projection_head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, projection_dim),
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_domains),
        )

    def _fuse(self, x: torch.Tensor) -> torch.Tensor:
        if self.branch_mode == "dual":
            acoustic_features = self.acoustic_branch(x)
            backbone_features = self.backbone_branch(x)
            if self.fusion_type == "agf":
                fused = self.fusion(acoustic_features, backbone_features)
            else:
                fused = torch.cat([acoustic_features, backbone_features], dim=1)
        elif self.branch_mode == "acoustic_only":
            fused = self.fusion(self.acoustic_branch(x))
        else:
            backbone_features = self.backbone_branch(x)
            fused = self.fusion(self.capacity_adapter(backbone_features))
        return self.fusion_norm(fused)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fused_features = self._fuse(x)
        logits = self.classifier(fused_features)
        contrastive_features = F.normalize(self.projection_head(fused_features), dim=1)
        domain_logits = self.domain_classifier(grad_reverse(fused_features, self.grl_alpha))
        return logits, fused_features, contrastive_features, domain_logits


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


class DGDBAFNet(AblationDGDBAFNet):
    """Main DG-DBAFNet model with ablation switches kept for controlled studies."""

    pass
