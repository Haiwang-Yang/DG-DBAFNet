#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import (  # noqa: E402
    collect_audio_items,
    split_audio_items,
    write_manifest,
)
from src.reproducibility import sha256_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate immutable file-level dataset manifests.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", default=str(PROJECT_ROOT / "manifests"))
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--limit_per_family", type=int, default=None)
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ctis_items = collect_audio_items(
        str(data_root),
        ["ctis"],
        limit_per_family=args.limit_per_family,
    )
    outputs = {}
    for split in ("train", "validation", "test"):
        selected = split_audio_items(
            ctis_items,
            split=split,
            seed=args.split_seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
        if not selected:
            raise RuntimeError(
                f"CTIS {split} split is empty. Increase available files or "
                "limit_per_family before using this manifest."
            )
        path = output_dir / f"ctis_{split}_seed{args.split_seed}.csv"
        write_manifest(path, selected, data_root)
        outputs[path.name] = {
            "files": len(selected),
            "sha256": sha256_file(path),
        }

    for dataset in ("chmusic", "folkmusic"):
        selected = collect_audio_items(
            str(data_root),
            [dataset],
            limit_per_family=args.limit_per_family,
        )
        if not selected:
            raise RuntimeError(f"{dataset} external manifest would be empty.")
        path = output_dir / f"{dataset}_external.csv"
        write_manifest(path, selected, data_root)
        outputs[path.name] = {
            "files": len(selected),
            "sha256": sha256_file(path),
        }

    metadata = {
        "split_seed": args.split_seed,
        "train_ratio": args.train_ratio,
        "validation_ratio": args.val_ratio,
        "clip_seconds": 5.0,
        "clip_hop_seconds": 5.0,
        "clip_overlap_seconds": 0.0,
        "manifests": outputs,
    }
    metadata_path = output_dir / "manifest_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
