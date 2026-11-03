# Reviewer Release Materials

This folder is used to organize materials requested during peer review:

1. code repository information;
2. dataset split files or split-file documentation;
3. code/data availability statements;
4. release checklist and exclusion rules.

## Recommended Review Strategy

Use a transparent staged-release strategy.

- During review: provide an anonymized or restricted-access repository link to editors/reviewers if the journal allows it.
- Upon acceptance/publication: make the full repository public.
- Do not redistribute raw audio files unless dataset licenses explicitly allow it.
- Do not claim that the full code is publicly reproducible if key files are intentionally withheld.

## Folder Layout

```text
review_release/
  README.md
  CODE_AVAILABILITY.md
  DATA_SPLITS.md
  RELEASE_CHECKLIST.md
  core_code/
  manifests/
```

## Suggested Public Repository Contents

The reviewer release currently includes:

- source code for the proposed method;
- environment file or dependency list;
- training and evaluation commands;
- immutable split manifests;
- metric/report generation scripts;
- README explaining dataset acquisition and expected directory layout.

The final public repository should not include:

- raw audio data;
- local absolute paths;
- private notes, draft manuscripts, reviewer comments, or submission files;
- trained checkpoints before publication, unless you choose to release them;
- feature caches or intermediate outputs.
