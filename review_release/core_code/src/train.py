import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import FAMILY_NAMES, LocalStackedFeatureDataset
from .evaluate import evaluate_dataset
from .metrics import compute_metrics
from .model import (
    BACKBONE_INITS,
    BRANCH_MODES,
    FUSION_TYPES,
    DGDBAFNet,
    count_parameters,
)
from .reproducibility import (
    make_generator,
    save_run_config,
    seed_worker,
    set_run_seed,
    sha256_file,
)


class DomainRobustSupConLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 0.07,
        strong_positive_weight: float = 2.0,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.temperature = temperature
        self.strong_positive_weight = strong_positive_weight
        self.eps = eps

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        domains: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if features.size(0) < 2:
            return features.new_tensor(0.0)
        labels = labels.contiguous().view(-1, 1)
        same_class = torch.eq(labels, labels.T).float().to(features.device)
        positive_weights = same_class
        if domains is not None:
            domains = domains.contiguous().view(-1, 1)
            different_domain = (~torch.eq(domains, domains.T)).float().to(features.device)
            positive_weights = same_class * (
                1.0 + (self.strong_positive_weight - 1.0) * different_domain
            )

        logits = torch.matmul(features, features.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True)[0].detach()
        logits_mask = torch.ones_like(same_class) - torch.eye(
            same_class.size(0),
            device=features.device,
        )
        positive_weights = positive_weights * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + self.eps)
        positive_weight_sum = positive_weights.sum(dim=1)
        valid = positive_weight_sum > 0
        if not torch.any(valid):
            return features.new_tensor(0.0)
        mean_log_prob_pos = (
            (positive_weights * log_prob).sum(dim=1)[valid]
            / positive_weight_sum[valid]
        )
        return -mean_log_prob_pos.mean()


def parse_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def save_csv(path: Path, rows: Iterable[Iterable[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)


def manifest_paths(args: argparse.Namespace) -> Dict[str, Path]:
    root = Path(args.manifest_dir)
    return {
        "train": root / f"ctis_train_seed{args.split_seed}.csv",
        "validation": root / f"ctis_validation_seed{args.split_seed}.csv",
        "ctis_test": root / f"ctis_test_seed{args.split_seed}.csv",
        "chmusic": root / "chmusic_external.csv",
        "folkmusic": root / "folkmusic_external.csv",
    }


def validate_manifests(paths: Dict[str, Path], validation_only: bool) -> None:
    required = ("train", "validation") if validation_only else tuple(paths)
    missing = [str(paths[name]) for name in required if not paths[name].exists()]
    if missing:
        raise FileNotFoundError(
            "Missing immutable manifests. Run scripts/generate_manifests.py first:\n"
            + "\n".join(missing)
        )


def make_dataset(
    args: argparse.Namespace,
    split: str,
    manifest_path: Path,
    pseudo_domain_augment: bool = False,
) -> LocalStackedFeatureDataset:
    return LocalStackedFeatureDataset(
        data_root=args.data_root,
        split=split,
        datasets=("ctis",),
        cache_dir=args.cache_dir,
        seed=args.split_seed,
        pseudo_domain_augment=pseudo_domain_augment,
        num_pseudo_domains=args.num_pseudo_domains,
        noise_scale_factor=args.noise_scale_factor,
        noise_scale_min=args.noise_scale_min,
        noise_scale_max=args.noise_scale_max,
        smoothing_kernel=args.smoothing_kernel,
        smoothing_blend=args.smoothing_blend,
        frequency_mask_ratio=args.frequency_mask_ratio,
        time_mask_ratio=args.time_mask_ratio,
        manifest_path=str(manifest_path),
    )


def make_loader(
    dataset: LocalStackedFeatureDataset,
    args: argparse.Namespace,
    shuffle: bool,
    seed_offset: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available() and not args.cpu,
        drop_last=shuffle and len(dataset) >= args.batch_size,
        worker_init_fn=seed_worker,
        generator=make_generator(args.run_seed + seed_offset),
        persistent_workers=args.num_workers > 0,
    )


def class_weights(dataset: LocalStackedFeatureDataset, device: torch.device) -> torch.Tensor:
    counts = np.zeros(len(FAMILY_NAMES), dtype=np.float32)
    for clip in dataset.clips:
        counts[dataset.audio_items[clip.audio_index].label] += 1
    weights = 1.0 / np.maximum(counts, 1.0)
    weights = weights / weights.sum() * len(weights)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_model(args: argparse.Namespace) -> DGDBAFNet:
    return DGDBAFNet(
        num_classes=len(FAMILY_NAMES),
        backbone=args.backbone,
        train_mode=0 if args.linear_probe else 1,
        branch_mode=args.branch_mode,
        fusion_type=args.fusion_type,
        num_domains=args.num_pseudo_domains,
        grl_alpha=args.grl_alpha,
        use_channel_attention=args.channel_attention,
        use_spatial_attention=args.spatial_attention,
        mixstyle_p=args.mixstyle_p,
        mixstyle_alpha=args.mixstyle_alpha,
        backbone_init=args.backbone_init,
        capacity_match_hidden=args.capacity_match_hidden,
        allow_weight_download=args.allow_weight_download,
    )


def run_training_epoch(
    model: nn.Module,
    loader: DataLoader,
    ce_criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    contrastive_criterion: Optional[nn.Module],
    domain_criterion: Optional[nn.Module],
    supcon_weight: float,
    pdadv_weight: float,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    y_true: List[int] = []
    y_pred: List[int] = []
    for batch in tqdm(loader, desc="Train", leave=False):
        inputs = batch["input"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        domains = batch["domain"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits, _, contrastive_features, domain_logits = model(inputs)
        loss = ce_criterion(logits, labels)
        if contrastive_criterion is not None and supcon_weight > 0:
            loss = loss + supcon_weight * contrastive_criterion(
                contrastive_features,
                labels,
                domains,
            )
        if domain_criterion is not None and pdadv_weight > 0:
            loss = loss + pdadv_weight * domain_criterion(domain_logits, domains)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * labels.size(0)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())
    return (
        total_loss / max(1, len(loader.dataset)),
        100.0 * float(accuracy_score(y_true, y_pred)),
    )


def evaluate_validation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, object]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            inputs = batch["input"].to(device, non_blocking=True)
            logits, _, _, _ = model(inputs)
            y_true.extend(batch["label"].tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
    return compute_metrics(y_true, y_pred, labels=(0, 1, 2, 3))


def evaluate_final(
    model: nn.Module,
    args: argparse.Namespace,
    paths: Dict[str, Path],
    device: torch.device,
    output_dir: Path,
) -> Dict[str, Dict[str, object]]:
    datasets = {
        "ctis_test": (
            make_dataset(args, "test", paths["ctis_test"]),
            ("ctis_full4", "shared3_unmasked", "shared3_masked_sensitivity"),
        ),
        "chmusic": (
            make_dataset(args, "all", paths["chmusic"]),
            ("external_present3_unmasked", "shared3_masked_sensitivity"),
        ),
        "folkmusic": (
            make_dataset(args, "all", paths["folkmusic"]),
            ("external_present3_unmasked", "shared3_masked_sensitivity"),
        ),
    }
    results: Dict[str, Dict[str, object]] = {}
    for offset, (name, (dataset, protocols)) in enumerate(datasets.items(), start=20):
        loader = make_loader(dataset, args, shuffle=False, seed_offset=offset)
        results[name] = evaluate_dataset(
            model,
            loader,
            device,
            output_dir / "evaluation",
            name,
            protocols,
        )
    return results


def train(args: argparse.Namespace) -> None:
    set_run_seed(args.run_seed, deterministic=args.deterministic)
    paths = manifest_paths(args)
    validate_manifests(paths, args.validation_only)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    run_dir = Path(args.output_dir) / args.experiment_id / f"seed_{args.run_seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_set = make_dataset(
        args,
        "train",
        paths["train"],
        pseudo_domain_augment=args.pseudo_domain_augment,
    )
    validation_set = make_dataset(args, "validation", paths["validation"])
    train_loader = make_loader(train_set, args, shuffle=True, seed_offset=0)
    validation_loader = make_loader(
        validation_set,
        args,
        shuffle=False,
        seed_offset=1,
    )

    model = build_model(args).to(device)
    total_params, trainable_params = count_parameters(model)
    ce_criterion = nn.CrossEntropyLoss(
        weight=class_weights(train_set, device) if args.weighted_ce else None
    )
    contrastive_criterion = (
        DomainRobustSupConLoss(
            temperature=args.supcon_temperature,
            strong_positive_weight=args.strong_positive_weight,
        )
        if args.supcon_weight > 0
        else None
    )
    domain_criterion = nn.CrossEntropyLoss() if args.pdadv_weight > 0 else None
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    config = vars(args).copy()
    config.update(
        {
            "device": str(device),
            "total_params": total_params,
            "trainable_params": trainable_params,
            "manifest_sha256": {
                name: sha256_file(path)
                for name, path in paths.items()
                if path.exists()
            },
        }
    )
    save_run_config(run_dir / "config.json", config)

    best_macro_f1 = -1.0
    best_epoch = -1
    checkpoint_path = run_dir / "best_checkpoint.pt"
    history = [["epoch", "train_loss", "train_accuracy", "validation_macro_f1"]]

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_training_epoch(
            model,
            train_loader,
            ce_criterion,
            optimizer,
            device,
            contrastive_criterion,
            domain_criterion,
            args.supcon_weight,
            args.pdadv_weight,
        )
        validation_macro_f1 = ""
        if epoch % args.validation_interval == 0 or epoch == args.epochs:
            validation_metrics = evaluate_validation(model, validation_loader, device)
            validation_macro_f1 = float(validation_metrics["macro_f1"])
            if validation_macro_f1 > best_macro_f1:
                best_macro_f1 = validation_macro_f1
                best_epoch = epoch
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "validation_macro_f1": best_macro_f1,
                        "config": config,
                    },
                    checkpoint_path,
                )
        history.append([epoch, train_loss, train_accuracy, validation_macro_f1])
        save_csv(run_dir / "history.csv", history)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    validation_metrics = evaluate_validation(model, validation_loader, device)
    final_results = {}
    if not args.validation_only:
        final_results = evaluate_final(model, args, paths, device, run_dir)

    summary = {
        "experiment_id": args.experiment_id,
        "split_seed": args.split_seed,
        "run_seed": args.run_seed,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_macro_f1,
        "validation_metrics": validation_metrics,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "backbone_init": args.backbone_init,
        "branch_mode": args.branch_mode,
        "fusion_type": args.fusion_type,
        "validation_only": args.validation_only,
        "final_results": final_results,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (run_dir / "COMPLETED.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "best_epoch": best_epoch,
                "best_validation_macro_f1": best_macro_f1,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Train one reproducible DG-DBAFNet reviewer experiment."
    )
    parser.add_argument("--experiment_id", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument(
        "--manifest_dir",
        default=str(project_root / "manifests"),
    )
    parser.add_argument(
        "--cache_dir",
        default=str(project_root / "feature_cache"),
    )
    parser.add_argument(
        "--output_dir",
        default=str(project_root / "outputs"),
    )
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--run_seed", type=int, default=42)
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--backbone_init", choices=BACKBONE_INITS, default="imagenet")
    parser.add_argument("--branch_mode", choices=BRANCH_MODES, default="dual")
    parser.add_argument("--fusion_type", choices=FUSION_TYPES, default="agf")
    parser.add_argument("--capacity_match_hidden", type=int, default=0)
    parser.add_argument("--linear_probe", action="store_true")
    parser.add_argument("--allow_weight_download", action="store_true")
    parser.add_argument("--channel_attention", type=parse_bool, default=True)
    parser.add_argument("--spatial_attention", type=parse_bool, default=True)
    parser.add_argument("--pseudo_domain_augment", type=parse_bool, default=True)
    parser.add_argument("--num_pseudo_domains", type=int, default=4)
    parser.add_argument("--noise_scale_factor", type=float, default=0.05)
    parser.add_argument("--noise_scale_min", type=float, default=0.005)
    parser.add_argument("--noise_scale_max", type=float, default=0.05)
    parser.add_argument("--smoothing_kernel", type=int, default=7)
    parser.add_argument("--smoothing_blend", type=float, default=0.3)
    parser.add_argument("--frequency_mask_ratio", type=float, default=0.1)
    parser.add_argument("--time_mask_ratio", type=float, default=0.1)
    parser.add_argument("--mixstyle_p", type=float, default=0.5)
    parser.add_argument("--mixstyle_alpha", type=float, default=0.1)
    parser.add_argument("--supcon_weight", type=float, default=0.1)
    parser.add_argument("--supcon_temperature", type=float, default=0.07)
    parser.add_argument("--strong_positive_weight", type=float, default=2.0)
    parser.add_argument("--pdadv_weight", type=float, default=0.05)
    parser.add_argument("--grl_alpha", type=float, default=1.0)
    parser.add_argument("--weighted_ce", type=parse_bool, default=True)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--validation_interval", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--validation_only", action="store_true")
    parser.add_argument("--deterministic", type=parse_bool, default=True)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
