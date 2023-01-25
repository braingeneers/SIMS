# **SIMS**: Scalable, Interpretable Modeling for Single-Cell RNA-Seq Data Classification

SIMS is a pipeline for building interpretable and accurate classifiers for intentifying any target on single-cell rna-seq data. The SIMS model is based on [TabNet](https://arxiv.org/abs/1908.07442), a self-attention based model specifically built for large-scale tabular datasets.

SIMS takes in a list of arbitrarily many expression matrices along with their corresponding target variables. The expression matrices may be AnnData objects with format `h5ad`, or `.csv`. 
They must be in the matrix form `cell x gene`, and NOT `gene x cell`, since our training samples are the transcriptomes of individual cells.

The data is formated like so:
- All matrices are cell x expression
- All label files contain a common column, known as the `class_label`, on which to train the model 
- `datafiles` and `labelfiles` are the absolute paths to the expression matrices and labels, respectively

A call to generate and train the SIMS model looks like the following:

```python 
from pytorch_lightning.loggers import WandbLogger
from scsims import SIMS

wandb_logger = WandbLogger(project=f"My Project", name=f"SIMS Model Training") # set up the logger to log data to Weights and Biases

sims = SIMS(adata=adata, labels_key='subclass_label') # set up the base class with anndata object and labels
sims.train() # Run the training loop
```

This will train the TabNet model on the given expression matrices with target variable given by the `class_label` column in each label file.