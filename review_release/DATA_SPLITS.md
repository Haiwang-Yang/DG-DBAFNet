# Dataset Split Files

## Split Policy Used By The Current Revision Experiments

The current revision experiment protocol is based on immutable manifest files generated before training.

- CTIS: train / validation / test split at original-audio-file level.
- Split seed: 42.
- CTIS ratio: 70% train, 10% validation, 20% test.
- ChMusic: external evaluation manifest.
- FolkMusic: external evaluation manifest.

Included manifest files:

```text
manifests/
  ctis_train_seed42.csv
  ctis_validation_seed42.csv
  ctis_test_seed42.csv
  chmusic_external.csv
  folkmusic_external.csv
  manifest_metadata.json
```

## Recommended Manifest Columns

Each CSV should use portable relative paths rather than local absolute paths.

```text
dataset,source,member,label,family,duration
```

- `source`: path relative to the dataset root.
- `member`: zip member name if the audio comes from an archive; empty otherwise.
- `label`: integer family label.
- `family`: readable family name.
- `duration`: original audio duration in seconds.

Do not expose private local absolute paths in public files.

## Reviewer-Facing Note

The split manifests identify which audio files belong to each experimental partition. Raw audio files are not redistributed with the repository because they belong to their original dataset providers.

## Integrity Check

The manifests were generated with split seed 42 and validated against the dataset layout used by the experiments.

- CTIS train: 3,467 recordings.
- CTIS validation: 493 recordings.
- CTIS test: 996 recordings.
- ChMusic external evaluation: 55 recordings.
- FolkMusic external evaluation: 11,966 recordings.
- CTIS train, validation, and test recording identifiers are mutually disjoint.
- SHA-256 checksums are stored in `manifests/manifest_metadata.json`.
