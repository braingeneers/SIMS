# Changelog

All notable changes to scsims are documented in this file. The format is
loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/).

## [4.0.0] — Unreleased

This is the long-overdue dependency-stack and API modernization release.
scsims 3.x had hard `==` pins on a 2023-era stack (numpy 1.24, pandas 2.0,
torch 1.13, lightning 2.0.2, torchmetrics 0.11) that became uninstallable
on Streamlit Cloud once `setuptools 81` removed `pkg_resources` and
Python 3.12+ wheels stopped existing for the pinned packages. v4 lifts
the entire stack to current versions and cleans up several latent bugs
that have been in the codebase since the 3.0 series.

See [`MIGRATION.md`](MIGRATION.md) for the v3 → v4 upgrade guide.

### Added

- **Real TabNet self-supervised pretraining.** `scsims/pretraining.py` was
  non-functional dead code in 3.x (`pretrain_model` had body `pass`,
  `_compute_metrics` referenced an undefined `metric` symbol). v4 ships a
  working implementation:
  - `scsims.SIMSPretrainer` — a Lightning module wrapping
    `pytorch_tabnet.tab_network.TabNetPretraining`. Implements TabNet's
    random feature obfuscation + variance-normalized reconstruction loss
    from the original paper.
  - `scsims.SIMS.pretrain(...)` — runs unsupervised pretraining on the
    same DataModule used for supervised training.
  - `scsims.SIMS.load_pretrainer(weights_path)` — loads a previously
    saved pretrainer checkpoint, for the two-process workflow where
    pretraining and fine-tuning happen in separate Python sessions.
  - `scsims.transfer_pretrained_weights(classifier, pretrainer)` —
    explicit helper that copies encoder + embedder weights from a
    pretrainer into a fresh `SIMSClassifier`. Mirrors
    pytorch-tabnet's own `load_weights_from_unsupervised`. Returns the
    number of tensors transferred for sanity-checking the warm start.
  - `SIMS.train()` automatically detects an attached pretrainer (either
    from `pretrain()` or `load_pretrainer()`) and warm-starts the
    classifier from its encoder weights before fitting.
- **`top_k` parameter on `SIMSClassifier.predict()` and `SIMS.predict()`.**
  Defaults to 3 (preserves prior behaviour). Pass `top_k=1` for the
  common "just give me the best label" case, or any positive integer up
  to the number of training classes. Output frame columns scale
  accordingly: `pred_0..pred_{top_k-1}` and `prob_0..prob_{top_k-1}`.
- **`grouped_features` parameter on `SIMSClassifier.__init__`.** Lets the
  user supply explicit gene-feature groupings to TabNet's group attention
  matrix instead of falling back to the one-feature-per-group default.
- **`scsims.contrib` subpackage** for optional integrations with extra
  dependencies. Currently houses the S3 upload callback (see below).
- **`pyproject.toml`** with PEP 621 metadata. Replaces `setup.py` and
  `setup.cfg`.
- **CI**: GitHub Actions workflow running pytest on Python 3.10, 3.11, 3.12.
- **Real test suite** under `tests/`: 10 pytest tests covering
  smoke (train/predict/explain), top-k, gene-alignment, pretraining,
  warm-start transfer, and an opt-in legacy-checkpoint regression test.

### Changed

#### Dependency stack (the unblocking change)

- `requires-python = ">=3.10"` (was 3.6 in classifiers, 3.9 in
  `python_requires`).
- All hard `==` pins replaced with lower bounds:
  - `numpy>=1.26` (was 1.24.3)
  - `pandas>=2.1` (was 2.0.1)
  - `scipy>=1.11` (was 1.10.1)
  - `scikit-learn>=1.3` (was 1.2.2)
  - `torch>=2.2` (was 1.13.1)
  - `lightning>=2.2` (was `pytorch-lightning==2.0.2`)
  - `torchmetrics>=1.0` (was 0.11.4)
  - `pytorch-tabnet>=4.1`
  - `anndata>=0.10` (was 0.9.1)
- Trimmed `requirements.txt` from 106 over-pinned packages down to the
  10 the package actually imports. The previous file looked like a
  `pip freeze` from someone's dev box and forced installation of unrelated
  things like `boto3`, `wandb`, `fastapi`, `lightning-cloud`,
  `appdirs`, `inquirer`, `croniter`, etc.
- `boto3` is now an **optional** `s3` extra: `pip install scsims[s3]`.
  Previously every install pulled it in unconditionally.
- `scanpy` is now a **test-only** `test` extra. The library never imports
  scanpy at runtime — it was only used by `scsims/tests.py`.

#### API standardization

- `predict()` now consistently returns columns named `pred_0/pred_1/...`
  and `prob_0/prob_1/...`. The number of columns scales with the new
  `top_k` parameter. Previously `top_k` was hard-coded to
  `min(3, num_classes)` in two unrelated places that had to stay in sync
  manually.
- The `train_metrics` and `val_metrics` torchmetrics dicts are now
  `MetricCollection` instances registered as submodules, so Lightning
  automatically moves them to the correct device. Removes the manual
  `.to(device)` pattern and the module-level `device =` constant.
- `SIMS.setup_trainer()` accepts `checkpoint_dir` and `monitor` keyword
  arguments. Previous code hard-coded `./sims_checkpoints` and
  `val_loss`.
- The `model_size` argument on `SIMS.setup_model()` now raises a clear
  `ValueError` with the list of valid presets instead of an `assert`
  whose error message contradicted the documented values.

#### Imports & module layout

- `import pytorch_lightning as pl` → `import lightning.pytorch as pl`
  everywhere. Both still work but the latter is canonical in
  `lightning>=2.x`.
- `from torchmetrics.functional.classification.stat_scores import
  _stat_scores_update` → public `from torchmetrics.functional import
  stat_scores`. The private import was removed in `torchmetrics>=1.0`.
- `UploadCallback` moved from `scsims.networking` to `scsims.contrib`.
  The old import path still works but emits a `DeprecationWarning`.

### Fixed

- **`pkg_resources` ImportError on Streamlit Cloud.** Root cause of the
  production app outage. The 3.x dependency stack pinned
  `lightning-utilities==0.8.0`, which still does the legacy `import
  pkg_resources`. `pkg_resources` was removed in `setuptools 81`. The
  modern `lightning-utilities>=0.10` no longer imports it.
- **`torch.load` weights_only crash on legacy checkpoints.** v4
  passes `weights_only=False` from the SIMS facade when loading
  checkpoints, which is required because SIMS checkpoints embed an
  `sklearn.LabelEncoder` and numpy arrays via Lightning's
  `save_hyperparameters()`. SIMS checkpoints are trusted artifacts
  produced by the user's own training run.
- **`pytorch-tabnet 4.x` `group_attention_matrix=[]` crash.** The
  default empty list triggers an `AttributeError` inside
  `EmbeddingGenerator`. v4 builds a proper identity-style group matrix
  via `pytorch_tabnet.utils.create_group_matrix` so each gene is its
  own group when no explicit grouping is supplied.
- **Mutation bug in `configure_optimizers()`.** Previous code did
  `self.optim_params.pop("optimizer")` and `self.scheduler_params.pop
  ("scheduler")`, which broke any second call (e.g. re-fitting in the
  same Python session). Now copies before popping.
- **`UploadCallback` AWS env-var crash at import time.** The old code
  did `os.environ["AWS_SECRET_ACCESS_KEY"]` in `__init__`, which raised
  `KeyError` from a fresh process if the env var wasn't set. v4's
  contrib version accepts explicit credentials and falls back to the
  standard boto3 credential chain.
- **`UploadCallback.on_train_end` double-join bug.** The previous code
  did `os.path.join(self.path, self.checkpoint_callback.best_model_path)`
  but `best_model_path` is already absolute, so the join produced
  paths like `model_checkpoints/model_checkpoints/...` that didn't
  exist on disk and made the upload silently fail.
- **`np.array` → `np.ndarray`** in the `SIMSClassifier.predict` type
  annotation. `np.array` is a function, not a type; the annotation was
  meaningless.
- **MIT vs GPL v2 metadata mismatch.** The `LICENSE` file is MIT but
  `setup.py` advertised `License :: OSI Approved :: GNU General Public
  License v2 (GPLv2)` to PyPI. Now correctly says MIT.
- **Wrong Python version classifiers.** `setup.py` listed Python 3.6,
  3.7, 3.8 in classifiers despite `python_requires=">=3.9"`. v4
  classifies 3.10/3.11/3.12.
- **Inconsistent `model_size` docstring.** The assertion message said
  the valid values were `'big' / 'medium' / 'small'`, but the actual
  accepted values were `'tall' / 'grande' / 'venti' / 'trenta'`. Fixed
  the message.

### Removed

- `scsims/tests.py` — moved to `tests/test_smoke.py` and ported to
  pytest.
- `tests/test_data.py`, `tests/test_model.py`, `tests/test_helpers.py`,
  `tests/tests.py` — dead since at least 2022. They imported from a
  `src/` directory that hasn't existed since the 3.0 split.
- `setup.py`, `setup.cfg` — replaced by `pyproject.toml`.
- `UploadCallback` is no longer re-exported from the top-level `scsims`
  namespace. Use `from scsims.contrib import UploadCallback` instead.
  The `scsims.networking` module remains as a deprecation shim for
  one release.
- Module-level `device = torch.device("cuda:0" if available else "cpu")`
  constants in `model.py` and `lightning_train.py`. Lightning manages
  device placement.
- The dead `_inference_device` attribute on `SIMSClassifier` (set
  once, never read).
