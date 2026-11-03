import csv
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import FAMILY_NAMES, evaluate_logits


def _batch_metadata(batch: Mapping[str, object], index: int) -> Dict[str, object]:
    clip_start = batch["clip_start"][index]
    if torch.is_tensor(clip_start):
        clip_start = float(clip_start.item())
    return {
        "dataset": batch["dataset"][index],
        "file_id": batch["file_id"][index],
        "sample_id": batch["sample_id"][index],
        "source": batch["source"][index],
        "member": batch["member"][index],
        "clip_start": clip_start,
    }


def predict_logits(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, object]]]:
    model.eval()
    labels: List[int] = []
    logits_rows: List[np.ndarray] = []
    metadata: List[Dict[str, object]] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predict", leave=False):
            inputs = batch["input"].to(device, non_blocking=True)
            logits, _, _, _ = model(inputs)
            batch_labels = batch["label"].cpu().numpy()
            batch_logits = logits.cpu().numpy()
            labels.extend(batch_labels.tolist())
            logits_rows.extend(batch_logits)
            metadata.extend(_batch_metadata(batch, index) for index in range(len(batch_labels)))
    return (
        np.asarray(labels, dtype=np.int64),
        np.asarray(logits_rows, dtype=np.float64),
        metadata,
    )


def save_predictions(
    path: Path,
    y_true: Sequence[int],
    logits: np.ndarray,
    metadata: Sequence[Mapping[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "file_id",
        "sample_id",
        "source",
        "member",
        "clip_start",
        "true_label",
        "true_family",
        *[f"logit_{name}" for name in FAMILY_NAMES],
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for label, score_row, sample in zip(y_true, logits, metadata):
            row = dict(sample)
            row["true_label"] = int(label)
            row["true_family"] = FAMILY_NAMES[int(label)]
            for family, score in zip(FAMILY_NAMES, score_row):
                row[f"logit_{family}"] = f"{float(score):.10g}"
            writer.writerow(row)


def save_protocol_metrics(
    output_dir: Path,
    dataset_name: str,
    y_true: np.ndarray,
    logits: np.ndarray,
    metadata: Sequence[Mapping[str, object]],
    protocols: Sequence[str],
) -> Dict[str, Dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Dict[str, object]] = {}
    for protocol in protocols:
        metrics, _, _, _ = evaluate_logits(
            y_true,
            logits,
            protocol,
            metadata=metadata,
        )
        results[protocol] = metrics
        path = output_dir / f"{dataset_name}_{protocol}_metrics.json"
        path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return results


def evaluate_dataset(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    dataset_name: str,
    protocols: Sequence[str],
) -> Dict[str, Dict[str, object]]:
    y_true, logits, metadata = predict_logits(model, loader, device)
    save_predictions(output_dir / f"{dataset_name}_predictions.csv", y_true, logits, metadata)
    return save_protocol_metrics(
        output_dir,
        dataset_name,
        y_true,
        logits,
        metadata,
        protocols,
    )
