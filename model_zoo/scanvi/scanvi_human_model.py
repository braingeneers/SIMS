import scvi
import scvi.data
import pandas as pd 
import numpy as np

import os
import pathlib 
import sys
import anndata as an
from sklearn.model_selection import train_test_split 
from sklearn.metrics import *
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from os.path import join, dirname, abspath
sys.path.append(join(dirname(abspath(__file__)), '..', 'src'))

from data import *
from lightning_train import *
from model import *
from torchmetrics.functional import *
from networking import download, list_objects

here = pathlib.Path(__file__).parent.resolve()

# Download training data
for file in ['human_labels.csv', 'human.h5ad']:
    print(f'Downloading {file}')

    if not os.path.isfile(file):
        download(
            remote_name=join('jlehrer', 'human_benchmark', file),
            file_name=join(here, file),
        )

# Set up the data for scVI
data = an.read_h5ad(join(here, 'human.h5ad'))
data.X = data.X.todense()

labels = pd.read_csv(join(here, 'human_labels.csv'), index_col="sample_name")

nan_indices = np.where(labels["subclass_label"].isna())[0]

labels = labels.dropna(subset=["subclass_label"])
X_clean = np.delete(data.X, nan_indices, axis=0)

# Set up new anndata where we only have non-NaN labels
clean_human = an.AnnData(
    X=X_clean,
    obs=labels,
    var=data.var
)

clean_human.obs["label"] = clean_human.obs["subclass_label"].values
clean_human.obs["label"] = pd.Series(labels["subclass_label"], dtype="category")
clean_human.obs = clean_human.obs.reset_index()

# Set up train/val/test split same as SIMS model 
indices = clean_human.obs.loc[:, 'subclass_label']
train, val = train_test_split(indices, test_size=0.2, random_state=42, stratify=indices)
train, test = train_test_split(train, test_size=0.2, random_state=42, stratify=train)

train_data = clean_human[train.index, :]
valid_data = clean_human[val.index, :]
test_data = clean_human[test.index, :]

# Train the scVI model 
train_data = train_data.copy()
scvi.model.SCVI.setup_anndata(train_data)
vae = scvi.model.SCVI(train_data, n_layers=2, n_latent=30, gene_likelihood="nb")

vae.train(
    early_stopping=True,
    max_epochs=5,
    early_stopping_patience=20,
)

# Train the scANVI model
lvae = scvi.model.SCANVI.from_scvi_model(
    vae,
    adata=train_data,
    labels_key="subclass_label",
    unlabeled_category="N/A", # All are labeled, so we ignore this 
)

lvae.train(
    max_epochs=5, 
    n_samples_per_label=50,
)

# Now get the test accuracy
# Also use a PyTorch logger to we can visualize the results 
from pl_lightning.loggers import WandbLogger 

logger = WandbLogger(
    project='scANVI Comparison',
    name='Human Model (Allen Brain Institute Data)'
)
preds = lvae.predict(test_data)
truth = test_data.obs['subclass_label'].values

acc = accuracy_score(preds, truth)
logger.log("accuracy", acc)

f1 = f1_score(preds, truth, average=None)
mf1 = np.nanmedian(f1)

logger.log("Median f1", mf1)