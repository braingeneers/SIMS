# **SIMS**: Scalable, Interpretable Modeling for Single-Cell RNA-Seq Data Classification

SIMS is a pipeline for building interpretable and accurate classifiers for identifying any target on single-cell rna-seq data. The SIMS model is based on [Tabnet](https://arxiv.org/abs/1908.07442), a Deep Learning model specifically built for large-scale tabular datasets.

SIMS takes in a list of arbitrarily many expression matrices along with their corresponding target variables. We assume the matrix form `cell x gene`, and NOT `gene x cell`, since our training samples are the transcriptomes of individual cells.

The code is run with `python`. To use the package, we recommend using a virtual environment such as [miniconda](https://docs.conda.io/en/latest/miniconda.html) which will allow you to install packages without harming your system `python`.  

## Installation

scsims requires **Python 3.10 or newer**. We recommend a clean virtual
environment.

```bash
# conda
conda create -n sims python=3.11
conda activate sims

# or venv
python3.11 -m venv .venv
source .venv/bin/activate
```

Install the package:

```bash
# Latest stable release from PyPI
pip install scsims

# Or directly from GitHub
pip install git+https://github.com/braingeneers/SIMS.git

# With the optional S3 model upload callback
pip install "scsims[s3]"

# For development (pytest, ruff, build)
pip install -e ".[dev]"
```

> **Upgrading from 3.x?** See [`MIGRATION.md`](MIGRATION.md). The most
> common change is updating `predict()` call sites to use the canonical
> `pred_0` / `prob_0` column names instead of the (always-broken) `first_pred`
> / `first_prob` references.

## Training and inference
The sims library uses a cell-by-gene matrix. This means our input data to the model should be 
an (M, N) matrix of M cells with expression levels across N different genes. The data should be log1p normalized before model training and model inference. 

To train a model, we can set up a SIMS class in the following way:

```python
from scsims import SIMS
from lightning.pytorch.loggers import WandbLogger
import scanpy as sc

logger = WandbLogger(offline=True)

adata = sc.read_h5ad('mydata.h5ad')
#Perform some light filtering
sc.pp.filter_cells(adata, min_genes=100)
sc.pp.filter_genes(adata, min_cells=3)
#Transform the data for model ingestion
sc.pp.normalize_total(adata)#Normalize counts per cell
sc.pp.log1p(adata) ### Logarithmizing the data
sc.pp.scale(adata) #Scale mean to zero and variance to 1

sims = SIMS(data=adata, class_label='class_label')
sims.setup_trainer(accelerator="gpu", devices=1, logger=logger)
sims.train()
```

This will set up the underlying dataloaders, model, model checkpointing, and everything else we need. Model checkpoints will be saved every training epoch. 

To load in a model to infer new cell types on an unlabeled dataset, we load in the model checkpoint, point to the label file that we originally trained on, and run the `predict` method on new data.

```python
sims = SIMS(weights_path='myawesomemodel.ckpt')# If the model has been trained on GPU move the weights to CPU, this is the case for our pretrained models
#SIMS(weights_path=checkpoint_path,map_location=torch.device('cpu'))

unlabeled_data = sc.read_h5ad('my/new/unlabeled.h5ad')
#Process the data the same way you processed the training data. For all our pretrained models we followed this steps.
sc.pp.filter_cells(unlabeled_data, min_genes=100)
sc.pp.filter_genes(unlabeled_data, min_cells=3)
#Transform the data for model ingestion
sc.pp.normalize_total(unlabeled_data)#Normalize counts per cell
sc.pp.log1p(unlabeled_data) ### Logarithmizing the data
sc.pp.scale(unlabeled_data) #Scale mean to zero and variance to 1
#Perform the predictions
cell_predictions = sims.predict(unlabeled_data, top_k=3)

# `cell_predictions` is a pandas DataFrame with one row per input cell
# and the following columns:
#   pred_0, pred_1, pred_2  -- the top-3 predicted cell type labels
#   prob_0, prob_1, prob_2  -- the corresponding softmax probabilities
#
# For "just give me the best label" use top_k=1:
top1 = sims.predict(unlabeled_data, top_k=1)
unlabeled_data.obs["sims_label"] = top1["pred_0"].values
unlabeled_data.obs["sims_confidence"] = top1["prob_0"].values
```

Finally, to look at the explainability of the model, we similarly run
```python
explainability_matrix, _labels = sims.explain('my/new/unlabeled.h5ad')  # also accepts an AnnData object
```

## Self-supervised pretraining (new in v4)

scsims 4.0 ships a working implementation of TabNet's self-supervised
pretraining (random feature obfuscation + reconstruction loss). The
pretrained encoder warm-starts a downstream supervised classifier,
which is useful when you have a large unlabeled corpus and a smaller
labeled fine-tuning set.

```python
from scsims import SIMS
import scanpy as sc

unlabeled = sc.read_h5ad("big_unlabeled_corpus.h5ad")
sims = SIMS(data=unlabeled, class_label="cell_type")  # label col is ignored at this stage

# Stage 1: unsupervised pretraining. Cell type labels are not used.
sims.pretrain(
    pretraining_ratio=0.2,
    accelerator="gpu",
    devices=1,
    max_epochs=50,
)

# Stage 2: supervised fine-tuning. SIMS automatically detects the
# attached pretrainer and warm-starts the classifier from its encoder
# weights before fitting.
labeled = sc.read_h5ad("smaller_labeled_set.h5ad")
sims = SIMS(data=labeled, class_label="cell_type")
sims.load_pretrainer("./sims_pretrain_checkpoints/best.ckpt")
sims.setup_trainer(accelerator="gpu", devices=1, max_epochs=20)
sims.train()
```

## Custom training jobs / logging
To customize the underlying `pl.Trainer` and SIMS model params, we can initialize the SIMS model like
```python
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from scsims import SIMS
import anndata as an

adata = an.read_h5ad("my_labeled_data.h5ad")  # can read h5 using anndata as well
wandb_logger = WandbLogger(project=f"My Project", name=f"SIMS Model Training") # set up the logger to log data to Weights and Biases

sims = SIMS(data=adata, class_label='class_label')
sims.setup_model(n_a=64, n_d=64, weights=sims.weights)  # weighting loss inversely proportional by label freq, helps learn rare cell types (recommended)
sims.setup_trainer(
    logger=wandb_logger,
    callbacks=[
        EarlyStopping(
            monitor="val_loss",
            patience=50,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ],
    max_epochs=100,
)
sims.train()
```
This will train the SIMS model on the given expression matrices with target variable given by the `class_label` column in each label file.

## Using SIMS inside github codespaces
If you are using SIMS only for predictions using an already trained model, github codespaces is the recommended way to use this tool. You can also use this pipeline to train it in smaller datasets as the computing services offered in codespaces are modest.
To use this tool in github codespaces start by forking the repo in your github account. Then create a new codespace with the SIMS repo as the Repository of choice.
Once inside the newly created environment pull the latest SIMS image:
```docker
docker pull jmlehrer/sims:latest
```
Run the docker container mounting the file folder containing datasets and model checkpoints to the filesystem:
```docker
docker run -it -v /path/to/local/folder:/path/in/container [image_name] /bin/bash
```
Run main.py to check if the installation has been completed. You can alter this file as shown above to perform the different tasks.
```bash
python main.py
```
