import pandas as pd 
import os
import sys
import pandas as pd
import numpy as np
from torch.utils.data import *
from tqdm import tqdm
import linecache 
from pytorch_tabnet.tab_model import TabNetClassifier
sys.path.append('../src/')
sys.path.append('..')

from src.models.lib.data import *
from src.helper import *

def map_cols_test():
    ref = ['a', 'b', 'c']
    curr = ['b', 'a', 'c', 'd'] 
    sample = np.array([1,2,3,4]) # Want --> [2,1,3]

    result = clean_sample(sample, ref, curr)
    desired = torch.from_numpy(np.array([2,1,3]))
    
    assert torch.equal(result, desired)

    ref = ['a', 'b', 'c']
    curr = ['c', 'd', 'b', 'a']

    sample = np.array(
        [[1,2,3,4],
         [5,6,7,8]]
    ) 
    # --> want [[4, 3, 1],
    #           [8, 7, 5]]

    res = clean_sample(sample, ref, curr)
    desired = torch.from_numpy(np.array([
        [4,3,1],
        [8,7,5]
    ]))
    
    assert torch.equal(res, desired)
    
def _test_first_n_samples(n, datafile, labelfile):
    data = GeneExpressionData(datafile, labelfile, 'Type', skip=3)
    cols = data.columns
    
    # Generate dict with half precision values to read this into my 16gb memory
    dtype_cols = dict(zip(cols, [np.float32]*len(cols)))
    
    data_df = pd.read_csv(datafile, nrows=2*n, header=1, dtype=dtype_cols) # Might need some extras since numerical index drops some values
    label_df = pd.read_csv(labelfile, nrows=n)

    similar = []
    for i in range(n):
        datasample = data[i][0]
        dfsample = torch.from_numpy(data_df.loc[label_df.loc[i, 'cell'], :].values).float()
        
        isclose = all(torch.isclose(datasample, dfsample))
        similar.append(isclose)
    
    print(f"First {n=} columns of expression matrix is equal to GeneExpressionData: {all(p for p in similar)}")

    assert (all(p for p in similar))

def test_datamodule():
    N = 50
    datafiles, labelfiles = list(INTERIM_DATA_AND_LABEL_FILES_LIST.keys()), list(INTERIM_DATA_AND_LABEL_FILES_LIST.values())
    datafiles = [os.path.join('..', 'data', 'interim', f) for f in datafiles]
    processed_labels = [os.path.join('..', 'data', 'processed/labels', f) for f in labelfiles]

    for datafile, labelfile in zip(datafiles, processed_labels):
        print(f'Testing {datafile=}')
        _test_first_n_samples(N, datafile, labelfile)

if __name__ == "__main__":
    map_cols_test()
    test_datamodule()

