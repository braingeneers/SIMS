# **SIMS**: Scalable, Interpretable Modeling for Single-Cell RNA-Seq Data Classification

SIMS is a pipeline for building interpretable and accurate classifiers for identifying any target on single-cell rna-seq data. The SIMS model is based on [TabNet](https://arxiv.org/abs/1908.07442), a self-attention based model specifically built for large-scale tabular datasets.

SIMS takes in a list of arbitrarily many expression matrices along with their corresponding target variables. The expression matrices may be AnnData objects with format `h5ad`, or `.csv`. 
They must be in the matrix form `cell x gene`, and NOT `gene x cell`, since our training samples are the transcriptomes of individual cells.

The data is formatted like so:
- All matrices are cell x expression
- All label files contain a common column, known as the `class_label`, on which to train the model 
- `datafiles` and `labelfiles` are the absolute paths to the expression matrices and labels, respectively

The code is run with `python`. To use the package, we recommend using a virtual environment such as [miniconda](https://docs.conda.io/en/latest/miniconda.html) which will allow you to install packages without harming your computer's system `python`.  

1. Create a new virtual environment with `conda create --name=<NAME> python=3.9`
2. Install the SIMS package with `pip install git+https://github.com/braingeneers/SIMS.git`
3. Set up and run the model training in a `.py` file. A tutorial on how to do so is shown below.

A call to generate and train the SIMS model could look like the following:

```python 
from scsims import SIMS

sims = SIMS(adata='my/data/file.h5ad', labels_key='class_label')
sims.train()
```

To customize the underlying `pl.Trainer` and Tabnet model params, we can initialize the SIMS model like 
```python 
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from scsims import SIMS

wandb_logger = WandbLogger(project=f"My Project", name=f"SIMS Model Training") # set up the logger to log data to Weights and Biases

sims = SIMS(adata=adata, labels_key='class_label')
sims.setup_model(n_a=64, n_d=64, weights=sims.weights)  # weighting loss by label freq
sims.setup_trainer(
    logger=wandb_logger,
    callbacks=[
        EarlyStopping(
            monitor="val_loss",
            patience=50,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ],
    num_epochs=100,
)
sims.train()
```
This will train the TabNet model on the given expression matrices with target variable given by the `class_label` column in each label file.
