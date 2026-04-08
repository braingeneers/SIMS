# Migrating from scsims 3.x to 4.0

This guide covers the breaking changes in scsims 4.0 and how to update
existing code.

## TL;DR

For most users, the only change you actually need to make is at the
**call site of `model.predict()`**. The previous code reached for
columns named `first_pred` / `first_prob` that the library never
actually produced — those calls were already broken; v4 just makes the
correct names easier to find.

```diff
- predictions = sims.predict(adata)
- best_label = predictions["first_pred"]
- best_score = predictions["first_prob"]
+ predictions = sims.predict(adata, top_k=1)
+ best_label = predictions["pred_0"]
+ best_score = predictions["prob_0"]
```

If you're loading existing `.ckpt` files: **no action required**.
v4 deliberately preserves the `SIMSClassifier.__init__` signature so
checkpoints saved by 3.x continue to load and forward-pass cleanly.

---

## 1. Dependency stack: install change

### Before

```text
scsims==3.0.4
```

### After

```text
scsims>=4.0
```

The old version pinned a 2023-era dependency stack that became
uninstallable on Python 3.12+ and on systems with `setuptools>=81`.
4.0 declares lower bounds against current torch (≥2.2), lightning (≥2.2),
torchmetrics (≥1.0), numpy (≥1.26), pandas (≥2.1), anndata (≥0.10), and
pytorch-tabnet (≥4.1). Your `pip install` should now resolve cleanly on
Python 3.10–3.13.

If you previously had `pip install scsims` in a requirements file, you
no longer get `boto3` for free — it's now an optional `s3` extra.

```bash
# core only
pip install scsims

# with the S3 upload callback
pip install "scsims[s3]"

# for development (pytest, ruff, build)
pip install "scsims[dev]"
```

## 2. `model.predict()` output columns

This is the most important behavioural change.

### Before

`predict()` was hard-coded to return `min(3, num_classes)` predictions
per cell, with column names `pred_0..pred_2` and `prob_0..prob_2`.
A lot of user code (and the deployed sims_app Streamlit demo) reached
for `first_pred` / `first_prob` instead — those names were never
actually produced; the calls were silently broken since the 3.0 split.

### After

`predict()` accepts an explicit `top_k: int = 3` parameter and
documents the canonical output column scheme.

```python
predictions = sims.predict(adata)
# columns: pred_0, pred_1, pred_2, prob_0, prob_1, prob_2

predictions = sims.predict(adata, top_k=1)
# columns: pred_0, prob_0

predictions = sims.predict(adata, top_k=5)
# columns: pred_0..pred_4, prob_0..prob_4

predictions = sims.predict(adata, top_k=999)
# columns capped at the number of training classes
```

If you have code that referenced `first_pred` / `first_prob`, change
it to `pred_0` / `prob_0`. There is no compatibility shim for these
names because they were never a real API.

## 3. Lightning Trainer API

scsims 4.0 uses `lightning>=2.2`. If you were already on the
`pytorch-lightning` 2.0.x line, almost nothing changes. The biggest
gotcha is the canonical import path:

```diff
- import pytorch_lightning as pl
+ import lightning.pytorch as pl
```

`pytorch_lightning` is still re-exported and works, but the
`lightning.pytorch` path is what the official docs reference.

Trainer arguments removed in PL 2.0 (and therefore not supported by
scsims 4.0):

- `Trainer(gpus=N)` → `Trainer(accelerator="gpu", devices=N)`
- `Trainer(auto_lr_find=True)` → use the `Tuner` API
- `Trainer(progress_bar_refresh_rate=...)` → use a `RichProgressBar`
  callback or pass `enable_progress_bar=False`

## 4. `SIMS.setup_trainer()` accepts `checkpoint_dir` and `monitor`

You no longer have to construct a `ModelCheckpoint` callback by hand
just to redirect the output directory.

### Before

```python
sims.setup_trainer(
    callbacks=[pl.callbacks.ModelCheckpoint(dirpath="./my_ckpts")],
    max_epochs=20,
)
```

### After

```python
sims.setup_trainer(
    checkpoint_dir="./my_ckpts",
    max_epochs=20,
)
```

You can still pass `callbacks=` to add custom callbacks; scsims will
just skip the auto-added `ModelCheckpoint` if you supply your own.

## 5. `model_size` argument now raises `ValueError` cleanly

### Before

```python
SIMS(...).setup_model(model_size="big")
# AssertionError: if specified, model_size must be one of 'big', 'medium', or 'small'
# (which was a lie — the accepted values were 'tall', 'grande', 'venti', 'trenta')
```

### After

```python
SIMS(...).setup_model(model_size="big")
# ValueError: model_size must be one of ['grande', 'tall', 'trenta', 'venti'], got 'big'
```

## 6. `UploadCallback` moved to `scsims.contrib`

The S3 model checkpoint upload helper has moved out of the core API.

### Before

```python
from scsims.networking import UploadCallback

cb = UploadCallback(desc="my_run")
# always uploads to s3://braingeneersdev/model_checkpoints/
# crashes at __init__ time if AWS_ACCESS_KEY_ID isn't set
```

### After

```python
from scsims.contrib import UploadCallback

cb = UploadCallback(
    desc="my_run",
    bucket="my-bucket",
    endpoint_url="https://s3.us-west-2.amazonaws.com",  # optional
    metric="val_macro_accuracy",
    mode="max",
)
```

The new version takes `bucket` and `endpoint_url` as explicit
arguments instead of hard-coding the braingeneersdev S3 bucket. AWS
credentials follow the standard boto3 credential chain (explicit
kwargs → environment variables → `~/.aws/credentials` → instance
profile).

The old `from scsims.networking import UploadCallback` import still
works, but emits a `DeprecationWarning` and is hard-coded to the
braingeneersdev defaults. **It will be removed in scsims 5.0.**

## 7. `pretraining.py` is now real

If you somehow had code calling into `scsims.pretraining` in the 3.x
era — first off, congratulations on the most niche scsims usage in
existence. The 3.x version was a non-functional stub: `pretrain_model`
had body `pass`, `_compute_metrics` referenced an undefined `metric`
symbol, and there was no working training loop.

v4 ships a real implementation. See [README.md](README.md) for the new
pretraining quickstart, or use the new public API directly:

```python
from scsims import SIMS

sims = SIMS(data=unlabeled_adata, class_label="cell_type")

# Stage 1: self-supervised pretraining (cell type labels are ignored)
sims.pretrain(
    max_epochs=50,
    accelerator="gpu",
    devices=1,
    pretraining_ratio=0.2,
)

# Stage 2: supervised fine-tuning, automatically warm-started from the
# pretrained encoder.
sims.train(max_epochs=20, accelerator="gpu", devices=1)
```

## 8. Loading legacy `.ckpt` files

**No action required** — v4 will load any `.ckpt` produced by scsims
3.x without modification, including the `MGE_cortex.ckpt`,
`Allen_human.ckpt`, etc. shipped with the public sims_app demo.

What changed under the hood: `torch>=2.6` flipped the default of
`torch.load(weights_only=...)` from `False` to `True`. SIMS checkpoints
embed an `sklearn.LabelEncoder` and numpy arrays via Lightning's
`save_hyperparameters()`, which the new default refuses to deserialize.
v4's `SIMS.__init__` passes `weights_only=False` from one carefully
scoped call site, so the rest of the application keeps the safer
torch.load default.

You will see one informational warning the first time you load an old
checkpoint:

```
InconsistentVersionWarning: Trying to unpickle estimator LabelEncoder
from version 1.2.2 when using version 1.8.0. ...
```

This is sklearn telling you that the LabelEncoder was pickled with an
older sklearn. It's cosmetic — the LabelEncoder API is stable enough
that the embedded class list still deserializes correctly. The MGE
cortex regression test in `tests/test_legacy_checkpoint.py` exercises
exactly this code path.

## 9. Things that did *not* change

If you were doing any of the following, your code keeps working
unchanged:

- `from scsims import SIMS` (top-level facade)
- `SIMS(data=adata, class_label="...")` (constructor for training)
- `SIMS(weights_path="...")` (constructor for inference)
- `sims.train()` / `sims.predict()` / `sims.explain()` /
  `sims.set_temperature()` (core methods, with the column-name caveat
  in §2)
- The inference-time gene-alignment / zero-inflation logic — still
  handles "fewer genes in inference data than training" and "more
  genes in inference data than training" automatically.
- `SIMSClassifier` `__init__` signature — every kwarg from 3.x is
  still accepted, in the same order, with the same defaults. New
  kwargs were added but only at the end.
- Lightning checkpoint format — see §8.
