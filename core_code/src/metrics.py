from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


FAMILY_NAMES = ("wind", "percussion", "plucked_string", "bowed_string")
FULL4_LABELS = (0, 1, 2, 3)
SHARED3_LABELS = (0, 2, 3)


@dataclass(frozen=True)
class EvaluationProtocol:
    name: str
    labels: Tuple[int, ...]
    include_true_labels: Optional[Tuple[int, ...]] = None
    mask_logits_to_labels: bool = False


PROTOCOLS: Mapping[str, EvaluationProtocol] = {
    "ctis_full4": EvaluationProtocol("ctis_full4", FULL4_LABELS),
    "external_present3_unmasked": EvaluationProtocol(
        "external_present3_unmasked",
        SHARED3_LABELS,
        include_true_labels=SHARED3_LABELS,
        mask_logits_to_labels=False,
    ),
    "shared3_unmasked": EvaluationProtocol(
        "shared3_unmasked",
        SHARED3_LABELS,
        include_true_labels=SHARED3_LABELS,
        mask_logits_to_labels=False,
    ),
    "shared3_masked_sensitivity": EvaluationProtocol(
        "shared3_masked_sensitivity",
        SHARED3_LABELS,
        include_true_labels=SHARED3_LABELS,
        mask_logits_to_labels=True,
    ),
}


def predictions_from_logits(
    logits: np.ndarray,
    allowed_labels: Optional[Sequence[int]] = None,
) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != len(FAMILY_NAMES):
        raise ValueError(f"Expected logits shaped [N, 4], got {logits.shape}.")
    if allowed_labels is None:
        return logits.argmax(axis=1)
    allowed = np.asarray(tuple(allowed_labels), dtype=np.int64)
    if allowed.size == 0:
        raise ValueError("allowed_labels cannot be empty.")
    return allowed[logits[:, allowed].argmax(axis=1)]


def filter_by_true_label(
    y_true: np.ndarray,
    logits: np.ndarray,
    include_labels: Optional[Sequence[int]],
    metadata: Optional[Sequence[Mapping[str, object]]] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[List[Mapping[str, object]]]]:
    if include_labels is None:
        return y_true, logits, list(metadata) if metadata is not None else None
    keep = np.isin(y_true, np.asarray(tuple(include_labels), dtype=np.int64))
    filtered_metadata = None
    if metadata is not None:
        filtered_metadata = [row for row, selected in zip(metadata, keep) if selected]
    return y_true[keep], logits[keep], filtered_metadata


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Sequence[int],
) -> Dict[str, object]:
    y_true_array = np.asarray(y_true, dtype=np.int64)
    y_pred_array = np.asarray(y_pred, dtype=np.int64)
    labels_array = np.asarray(tuple(labels), dtype=np.int64)
    if y_true_array.size == 0:
        raise ValueError("Cannot evaluate an empty target array.")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_array,
        y_pred_array,
        labels=labels_array,
        average=None,
        zero_division=0,
    )
    macro_precision = float(np.mean(precision))
    macro_recall = float(np.mean(recall))
    macro_f1 = float(np.mean(f1))
    class_metrics = {
        FAMILY_NAMES[int(label)]: {
            "precision": 100.0 * float(p),
            "recall": 100.0 * float(r),
            "f1": 100.0 * float(score),
            "support": int(n),
        }
        for label, p, r, score, n in zip(labels_array, precision, recall, f1, support)
    }
    return {
        "accuracy": 100.0 * float(accuracy_score(y_true_array, y_pred_array)),
        "macro_precision": 100.0 * macro_precision,
        "macro_recall": 100.0 * macro_recall,
        "macro_f1": 100.0 * macro_f1,
        "labels": labels_array.tolist(),
        "class_metrics": class_metrics,
        "confusion_matrix": confusion_matrix(
            y_true_array,
            y_pred_array,
            labels=labels_array,
        ).tolist(),
        "absent_class_predictions": int(np.sum(~np.isin(y_pred_array, labels_array))),
        "n_samples": int(y_true_array.size),
    }


def evaluate_logits(
    y_true: Sequence[int],
    logits: np.ndarray,
    protocol_name: str,
    metadata: Optional[Sequence[Mapping[str, object]]] = None,
) -> Tuple[Dict[str, object], np.ndarray, np.ndarray, Optional[List[Mapping[str, object]]]]:
    if protocol_name not in PROTOCOLS:
        raise ValueError(f"Unknown protocol {protocol_name!r}.")
    protocol = PROTOCOLS[protocol_name]
    y_true_array = np.asarray(y_true, dtype=np.int64)
    logits_array = np.asarray(logits, dtype=np.float64)
    y_true_array, logits_array, filtered_metadata = filter_by_true_label(
        y_true_array,
        logits_array,
        protocol.include_true_labels,
        metadata,
    )
    allowed = protocol.labels if protocol.mask_logits_to_labels else None
    y_pred = predictions_from_logits(logits_array, allowed_labels=allowed)
    metrics = compute_metrics(y_true_array, y_pred, protocol.labels)
    metrics["protocol"] = protocol.name
    metrics["logit_masking"] = protocol.mask_logits_to_labels
    return metrics, y_true_array, y_pred, filtered_metadata
