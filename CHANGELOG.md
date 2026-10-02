# Changelog

Notable changes to TrainStudio. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/) — before 1.0.0, a minor release may change
behaviour.

## [Unreleased]

### Added
- Pause and resume. **⏸ Pause** stops a run when its epoch ends and **⏹ Stop now** stops it
  after the batch; a stopped or failed run can then be resumed from the Training page or
  the Dashboard, from the end of its last finished epoch, with results identical to an
  uninterrupted run. Every epoch writes `checkpoints/resume.pt`; `runner.py --resume`
  continues from it.
- Explanations on the Inference page: Grad-CAM, LIME and SHAP for any class, side by side
  if wanted, for classification and 2D segmentation — ViT, Swin and Ultralytics classifiers
  included. Needs the new `lime` and `shap` requirements.

### Fixed
- `seed` now reaches the augmentation: albumentations and MONAI transforms drew from
  generators seeded by the OS, so two runs with the same seed differed, and every
  DataLoader worker repeated the same augmentation sequence.
- Mask2Former checkpoints could not be loaded on the Inference page, and Grad-CAM picked a
  wrong layer on plain ViTs and torchvision's Swin.

## [0.1.0] — 2026-10-02

The first versioned release, licensed under the Apache License 2.0.

### Added
- Delete runs from the Dashboard, from the Training page once a run has ended, and from
  Results — one at a time or every ticked run at once. A run that is still training is
  never deleted.
- Regularisation & stability settings: dropout, stochastic depth, EMA and its decay,
  gradient clipping and layer-wise decay, plus Nesterov momentum, the step schedule,
  `torch.compile` and keeping the last checkpoint. Each model offers only what its
  library can apply, starting from that library's own default.
- Presets: save the current settings under a name and apply them, or any past run's
  configuration, to another model — the training recipe always, model-specific values
  only on request, machine-specific values never.
- A detailed dataset analysis (the Dataset page's Analysis tab and
  `check_dataset.py --analyze`): sizes and aspect ratios, channel modes and bit depth,
  formats, unreadable files, duplicates across splits, near-uniform images, channel
  statistics, and mask problems for segmentation.
- Standardising image sizes: a copy of the dataset with one size for every image
  (letterbox, resize or centre crop), written next to the original, which is never
  changed.
- A dialog explaining a dataset whose images lie directly in the split folders.
- The version, shown in the sidebar and recorded in each run's `env.json` and log.

### Fixed
- After a restart, a browser tab that reconnected to a page other than the Dashboard
  could start the app without its theme and navigation, for every session.
- The monitored metric defaulted to plain accuracy (or Dice) instead of balanced
  accuracy (or mean Dice), even on imbalanced datasets.
- A run that crashed, or died before training began, could show as running or queued
  forever.
- Overwriting a run left the previous attempt's logs, checkpoints and report in place.
- Ultralytics ignored "pretrained" off, and downloaded weights into the working
  directory instead of the cache.
- The Dashboard's memory reading on Apple GPUs always said 0 GB, and the first page
  load took several seconds checking the installed packages.
- Training and validation curves swapped colours between charts.

[0.1.0]: https://github.com/berkay-durmus/trainstudio/releases/tag/v0.1.0
