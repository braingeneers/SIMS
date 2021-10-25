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

from src.models.lib.lightning_train import generate_trainer 

trainer, model, data = generate_trainer(
    datafiles=['cortical_cells.csv', 'cortical_cells_2.csv', 'external/cortical_cells_3.h5ad'], # Notice we can mix and match file types
    labelfiles=['l1.csv', 'l2.csv', 'l3.csv'],
    class_label='cell_state', # Train to predict cell state!
    batch_size=4,
)

trainer.fit(model, datamodule=data)
```

This will train a derivation of the TabNet model on the given expression matrices with target variable given by the `class_label` column in each label file.