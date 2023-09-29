from scanpy.datasets import blobs 
from scsims import SIMS 
import numpy as np
import os 
import pytorch_lightning as pl 

def test_inference_more_genes_than_train():
    traindata = blobs(n_variables=10, n_observations=100)
    testdata = blobs(n_variables=20, n_observations=100)
    
    # make data 32-bit
    traindata.X = traindata.X.astype(np.float32)
    testdata.X = testdata.X.astype(np.float32)

    model = SIMS(data=traindata, class_label="blobs")
    model.train(max_epochs=2)

    predictions = model.predict(testdata)
    assert len(predictions) == 100
    
def test_inference_less_genes_than_train():
    traindata = blobs(n_variables=20, n_observations=100)
    testdata = blobs(n_variables=10, n_observations=100)
    
    traindata.obs["blobs"] = traindata.obs["blobs"].apply(lambda x: f"label_{x}")
    # make data 32-bit
    traindata.X = traindata.X.astype(np.float32)
    testdata.X = testdata.X.astype(np.float32)

    model = SIMS(data=traindata, class_label="blobs")
    model.train(max_epochs=2)

    predictions = model.predict(testdata, num_workers=0)
    assert len(predictions) == 100

def test_inference_decoded():
    traindata = blobs(n_variables=8, n_observations=100)  # has labels 0,..7 as str
    testdata = blobs(n_variables=10, n_observations=100)
    traindata.obs["blobs"] = traindata.obs["blobs"].apply(lambda x: f"label_{x}")
    traindata.X = traindata.X.astype(np.float32)
    testdata.X = testdata.X.astype(np.float32)

    model = SIMS(data=traindata, class_label="blobs")
    model.setup_trainer(callbacks=[pl.callbacks.ModelCheckpoint(dirpath="test_checkpoints")], max_epochs=2)
    model.train()

    # load in model to check if caching works
    checkpoint_path = os.path.join("test_checkpoints", os.listdir("test_checkpoints")[0])
    model = SIMS(weights_path=checkpoint_path)
    predictions = model.predict(testdata, num_workers=0)

    predicted_labels = predictions["first_pred"].unique()
    assert all(x.startswith("label_") for x in predicted_labels)

if __name__ == "__main__":
    test_inference_more_genes_than_train()
    test_inference_less_genes_than_train()
    test_inference_decoded()