# Core Code For Reviewer Inspection

This folder contains the core implementation files for the proposed DG-DBAFNet experiments.

## Main Files

```text
src/model.py
```

Defines the proposed network:

- `DGDBAFNet`, the main model used in the manuscript;
- acoustic attention branch;
- ImageNet-pretrained ResNet branch;
- adaptive gated fusion;
- classifier head;
- supervised contrastive projection head;
- gradient-reversal pseudo-domain classifier.

```text
src/train.py
```

Defines the training pipeline:

- manifest-based dataset loading;
- weighted cross-entropy;
- domain-robust supervised contrastive loss;
- pseudo-domain adversarial loss;
- validation-based checkpoint selection;
- final CTIS, ChMusic, and FolkMusic evaluation.

```text
src/dataset.py
src/features.py
```

Define audio loading, segmentation, MFCC/CQT/Chroma extraction, stacked `3 x 128 x 128` feature construction, feature caching, and pseudo-domain transforms.

```text
src/metrics.py
src/evaluate.py
src/plot.py
```

Define unified metrics, classification reports, confusion matrices, evaluation summaries, and plots.

```text
scripts/generate_manifests.py
scripts/check_manifests.py
```

Generate and validate immutable dataset split manifests.

## What Is Not Included

This reviewer-facing core-code folder intentionally excludes:

- raw audio data;
- trained checkpoints;
- feature caches;
- experiment output folders;
- local logs;
- draft manuscript files.

## Example Command

The commands below assume that the current directory is `core_code/` and that
the conda environment used for the revision experiments has been activated:

```bash
conda activate yue
```

```bash
python -m src.train \
  --experiment_id dg_dbafnet_main \
  --data_root /path/to/datasets \
  --manifest_dir ../manifests \
  --output_dir ./outputs \
  --cache_dir ./feature_cache
```

The exact dataset files must be obtained from their original providers. The
version-locked dependencies in `requirements.txt` were verified in the `yue`
conda environment. The immutable split manifests are stored in
`../manifests/`; use a different location only by passing it explicitly through
`--manifest_dir`.
