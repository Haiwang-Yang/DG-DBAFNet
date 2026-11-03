#!/usr/bin/env python3
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import FAMILY_NAMES, audio_file_id, read_manifest  # noqa: E402


def describe(items):
    counts = Counter(item.family for item in items)
    return {
        "files": len(items),
        "families": {family: counts.get(family, 0) for family in FAMILY_NAMES},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate immutable dataset manifests.")
    parser.add_argument("--manifest_dir", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--split_seed", type=int, default=42)
    args = parser.parse_args()

    names = {
        "train": f"ctis_train_seed{args.split_seed}.csv",
        "validation": f"ctis_validation_seed{args.split_seed}.csv",
        "test": f"ctis_test_seed{args.split_seed}.csv",
        "chmusic": "chmusic_external.csv",
        "folkmusic": "folkmusic_external.csv",
    }
    datasets = {
        name: read_manifest(args.manifest_dir / filename, args.data_root)
        for name, filename in names.items()
    }
    split_ids = {
        name: {audio_file_id(item) for item in datasets[name]}
        for name in ("train", "validation", "test")
    }
    if not split_ids["train"].isdisjoint(split_ids["validation"]):
        raise RuntimeError("CTIS train and validation manifests overlap.")
    if not split_ids["train"].isdisjoint(split_ids["test"]):
        raise RuntimeError("CTIS train and test manifests overlap.")
    if not split_ids["validation"].isdisjoint(split_ids["test"]):
        raise RuntimeError("CTIS validation and test manifests overlap.")

    for external in ("chmusic", "folkmusic"):
        labels = {item.label for item in datasets[external]}
        if 1 in labels:
            raise RuntimeError(f"{external} unexpectedly contains mapped percussion.")
        if labels != {0, 2, 3}:
            raise RuntimeError(
                f"{external} expected labels {{0,2,3}}, found {sorted(labels)}."
            )

    report = {
        "status": "valid",
        "split_seed": args.split_seed,
        "ctis_file_level_disjoint": True,
        "external_label_space": [0, 2, 3],
        "datasets": {name: describe(items) for name, items in datasets.items()},
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
