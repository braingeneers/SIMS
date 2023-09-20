import anndata as an 
import scanpy as sc 
from scsims import SIMS 
import torch

device = "cuda" if torch.cuda.is_available() else None
dataset = sc.datasets.pbmc3k_processed()

sims = SIMS(data=dataset, class_label="louvain")
sims.setup_trainer(devices=1, max_epochs=10, accelerator="gpu" if device else "cpu")
sims.train()

predictions = sims.predict(dataset)
