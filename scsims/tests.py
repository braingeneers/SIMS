from scanpy.datasets import blobs 
from scsims import SIMS 
import numpy as np


def test_inference_more_genes_than_train():
    traindata = blobs(n_variables=10, n_observations=100)
    testdata = blobs(n_variables=20, n_observations=100)
    
    # make data 32-bit
    traindata.X = traindata.X.astype(np.float32)
    testdata.X = testdata.X.astype(np.float32)

    # model = SIMS(data=traindata, class_label="blobs")
    # model.train(max_epochs=2)
    model = SIMS(weights_path="lightning_logs/version_0/checkpoints/epoch=1-step=4.ckpt", data=testdata, class_label="blobs")

    predictions = model.predict(testdata)
    assert len(predictions) == 100

def test_inference_less_genes_than_train():
    traindata = blobs(n_variables=20, n_observations=100)
    testdata = blobs(n_variables=10, n_observations=100)
    
    # make data 32-bit
    traindata.X = traindata.X.astype(np.float32)
    testdata.X = testdata.X.astype(np.float32)

    # model = SIMS(data=traindata, class_label="blobs")
    # model.train(max_epochs=2)
    model = SIMS(weights_path="lightning_logs/version_0/checkpoints/epoch=1-step=4.ckpt", data=testdata, class_label="blobs")

    predictions = model.predict(testdata)
    assert len(predictions) == 100

if __name__ == "__main__":
    test_inference_more_genes_than_train()