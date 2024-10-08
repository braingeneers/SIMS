import anndata as an 
import scanpy as sc 
from scsims import SIMS 
import torch


def main():
    device = "cuda" if torch.cuda.is_available() else None
    dataset = sc.datasets.pbmc3k_processed()

    sims = SIMS(data=dataset, class_label="louvain")
    sims.setup_trainer(devices=1, max_epochs=2, accelerator="gpu" if device else "cpu")
    sims.train()

    #Remove 100 genes from the dataset
    print(dataset.shape)
    dataset = dataset[:, 100:]
    print(dataset.shape)
    predictions = sims.predict(dataset)
    print(predictions)

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()