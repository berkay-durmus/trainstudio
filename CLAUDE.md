# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

TrainStudio is a Streamlit app for training image classification and 2D/3D segmentation
models (timm, torchvision, smp, HF transformers, Ultralytics, MONAI) on datasets read in
place from the local filesystem. The README is detailed and is the source of truth for
deployment, `.env` variables, dataset layouts and troubleshooting.

## Commands

Local development needs Python **3.11** (not 3.12/3.13 — MONAI/SimpleITK wheels). Install
PyTorch first, matched to the hardware, then `pip install -r requirements.txt`
(or `conda env create -f environment.yml && conda activate trainui`).

```bash
streamlit run app.py                                   # run the UI (localhost:8501)
python runner.py --config <run_dir>/config.json        # run a training job headless; exit 0 ok · 1 error · 2 stopped
python scripts/check_dataset.py <path>                 # how a dataset path is interpreted; exit 0 if usable
python scripts/make_dummy_dataset.py --out /tmp/ts_data --kinds cls seg dicom seg3d

python tests/ui/run_all.py                             # whole UI suite (builds fixtures first if missing)
python tests/ui/run_all.py t4                          # one part (t1…t9)
python tests/ui/run_all.py --rebuild                   # regenerate fixtures
python tests/ui/t4_flow.py                             # run one part's script directly (fixtures must exist)
```

There is no linter/formatter config and no pytest — the tests are plain scripts. Each
`tests/ui/t*.py` records checks through `harness.record()` and exits via
`harness.summary(...)`, which prints `UI TEST …: N/M passed`; `run_all.py` parses that line.
Fixtures (synthetic datasets plus two short real training runs) live in `.ui-fixtures/`
(override with `TS_UI_FIXTURES`); the first build trains models and takes minutes.

Docker: `make build`, `make up`, `make logs`, `make gpu-check`, `make proxy-up`, `make help`
for the rest. After changing mounts or env, `docker compose up -d --force-recreate`
(`restart` is not enough).

## Architecture

**UI and training are separate processes.** The Streamlit side never trains. It builds a
`RunConfig` (`core/schemas.py`, pydantic), writes it to `<run_dir>/config.json`, and
`core/launcher.py` `Popen`s `runner.py`. The runner talks back only through files in the
run directory, whose names are fixed in `core.schemas.Layout`:

- `events.jsonl`: append-only event stream (`core/events.py`, `EventWriter` / event names in `E`).
  The Training page tails it from a byte offset in a ~2 s fragment.
- `state.json`: atomically rewritten status / PID / latest epoch.
- `STOP`: written by the UI. The trainer checks for it between batches, saves `last.pt` and exits with status 2.
- `runner.out`: the process's stdout/stderr, where crashes before `fit()` show up.
- `PAUSE`: written by the UI. The trainer checks it once per epoch and stops after writing
  that epoch's `checkpoints/resume.pt` — the full training state (optimizer, scheduler,
  EMA, history, RNG). `launcher.resume_run` restarts `runner.py --resume` in the same run
  directory, appending to the same event stream; `runs.resume_info` says whether a run can be.

So `runner.py` and everything under `trainers/`, `metrics/` and `data/` must not import
Streamlit or `ui/`. `config.json` is the single source of truth for a run.

**Trainer dispatch.** `runner.build_trainer` picks a trainer class from
`(dataset.task, model.backend)`. Ultralytics takes over for any task. `trainers/base.py`
`BaseTrainer` owns the loop: optimizer/scheduler, AMP, EMA, freezing, early stopping, the
stop signal, event emission, checkpoints, metric tables and the final report. Subclasses
implement `build_model`, `build_data` and `new_metrics`, and optionally override
`forward_batch`, `compute_loss` and `save_preview`. A new backend needs a `Backend` enum
value, registry entries, a trainer, a branch in `build_trainer`, and checkpoint-loading
support in `export/inference.py` (its `model_logits` reads every kind of model output; LIME
and SHAP in `export/explain.py` see a model only through `ScoreFn`, so a new backend needs
nothing there unless Grad-CAM cannot find its layer). Build every DataLoader with
`**self.loader_source(split, dataset)`: the training split then draws its order and a
per-sample augmentation seed from torch's RNG (`data/seeding.py`), which is what makes
`hp.seed` reproducible and a resumed run identical to an uninterrupted one.

**Model catalogue and recommendations.** `core/registry.py` is a static list of `ModelSpec`
whose `arch` is passed straight to the backend library. Each spec's `rec` dict feeds
`core/recommend.py`, which combines it with dataset scan stats and `core/hardware.py`
detection into hyperparameter values plus a human-readable reason for each. On the Settings
page, fields the user edits are tracked (`K_TOUCHED`) and later recommendations never
overwrite them.

**Wizard state.** Pages (`views/0_…6_*.py`, wired up via `st.navigation` in `app.py`) pass
data to each other only through the `flow.*` keys and accessors in `ui/state.py`. Pages
should not read or write `st.session_state` directly. Never name the page folder `pages/`:
Streamlit then auto-discovers it, and a cold server whose first request is a deep link
(e.g. `/Training`) runs the page without `app.py`, so the theme and grouped navigation
are lost for every session until someone opens `/`. The folder picker (`ui/dir_picker.py`)
keeps its own keys, `_dp::<key>::<name>`.

**Backend capabilities.** `core/capabilities.py` says which settings each (backend, arch)
can apply and under which keyword — dropout is `drop_rate` in timm, `decoder_dropout` in smp
FPN, `dropout_prob` in MONAI SegResNet — and with what library default. The table was built
by constructing each catalogue model and finding the value in its layers, because some
constructors accept a keyword only to ignore it (torchvision ConvNeXt `dropout`). Trainers
pass `model_kwargs(cfg)`, which sends only values that differ from the library default, and
nothing at all for non-timm backends in a `config.json` older than `CONFIG_VERSION` 2.
`core/recommend.py` starts these fields from the library default, and the Settings page's
`field()` disables any field `supports()` rejects. Add a model → add its knobs here, measured.

**Presets.** `core/presets.py` stores hp + aug as JSON under `TRAINSTUDIO_HOME/presets/`; a
past run's `config.json` works as one too. `apply_preset` copies the recipe, copies
model-specific fields (`MODEL_SPECIFIC`) only on request, never hardware fields, and skips
task-specific or unsupported ones, returning the reasons.

**Datasets.** `data/spec.py` defines the single canonical layout and the alias/case-insensitive
folder-name matching, and also recognises YOLO's transposed `images/train` order. `data/scan.py`
validates a dataset and computes its stats. Datasets are never modified. A train-only dataset
gets a `splits.json` at its root, YOLO label conversion goes to a cache, and an optional
`dataset.yaml` holds the task, modality, classes and CT window. Path input is sanitised
(quotes, `file://`, `~`, `$VARS`) before use, and permission errors are reported, not raised
(`core/paths.py`). `data/analysis.py` (the Dataset page's Analysis tab, `ui/dataset_tools.py`)
reads every header and samples pixels; `data/standardize.py` writes a uniform-size copy that
mirrors the source path for path (so `splits.json` carries over) into a sibling folder marked
by `standardization.json` — the only feature that writes image data, and never in place.

## Testing notes

- Tests drive the real pages with `streamlit.testing.v1.AppTest.from_file` (`harness.run_page`)
  with the working directory set to the project root. Nothing is mocked.
- Assert on rendered elements (`harness.texts`, `tables`, `json_blocks`, `ss_get`) rather
  than raw page text (see recent commit "Assert the catalogue filters on the rendered cards").
- AppTest cannot reach `st.data_editor` or `st.switch_page` navigation, so those paths are
  tested by calling the underlying functions directly.

## Deployment gotchas

- In Docker, host directories are mounted at the **same path** inside the container. Keep
  that identity when adding mounts (`/srv/x:/srv/x`).
- `BASIC_AUTH_HASH` in `.env` needs every `$` doubled for Compose. Use
  `scripts/set_proxy_password.sh`, don't edit it by hand.
- `TORCH_VERSION` and `TORCHVISION_VERSION` must be a matching pair (2.5.1↔0.20.1, 2.6↔0.21, 2.7↔0.22).
