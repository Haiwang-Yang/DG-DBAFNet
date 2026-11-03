# Release Checklist

## Include

- README with method overview and commands.
- Source code for DG-DBAFNet and baselines.
- Dependency file, such as `environment.yml` or `requirements.txt`.
- Dataset directory template.
- Immutable split manifests.
- Evaluation scripts and metric definitions.
- License or usage notice.

## Exclude

- `data/`
- raw audio files
- feature caches
- trained checkpoints before publication
- `.pytest_cache/`
- `__pycache__/`
- local logs and TensorBoard runs
- draft manuscripts
- reviewer letters
- private notes
- local absolute paths

## Before Sharing A Link

1. Search for local absolute paths.
2. Search for private filenames or draft notes.
3. Confirm the README wording matches the actual access level.
4. Confirm split files use relative paths.
5. Confirm no raw audio files or checkpoints are accidentally included.

