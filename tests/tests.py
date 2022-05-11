import pandas as pd 
import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import *
from tqdm import tqdm
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
    
def _test_first_n_samples(n, datafile, labelfile, class_label='Type', index_col='cell'):
    data = GeneExpressionData(
        datafile, 
        labelfile, 
        class_label, 
        skip=3,
        index_col=index_col,
    )
    cols = data.columns
    
    # Generate dict with half precision values to read this into my 16gb memory
    dtype_cols = dict(zip(cols, [np.float32]*len(cols)))
    
    data_df = pd.read_csv(datafile, nrows=2*n, header=1, dtype=dtype_cols) # Might need some extras since numerical index drops some values
    label_df = pd.read_csv(labelfile, nrows=n)

    similar = []
    for i in range(n):
        datasample = data[i][0]

        dfsample = torch.from_numpy(data_df.loc[label_df.loc[i, index_col], :].values).float()
        isclose = all(torch.isclose(datasample, dfsample))
        similar.append(isclose)
    
    print(f"First {n=} columns of expression matrix is equal to GeneExpressionData: {all(p for p in similar)}")

    assert (all(p for p in similar))

def test_split(n, datafile, labelfile, class_label='Type', index_col='cell'):
    from sklearn.model_selection import train_test_split 
    # 2*n is usually enough to capture differences in index_col

    label_df = pd.read_csv(labelfile, nrows=2*n)
    data_df = pd.read_csv(datafile, nrows=2*n, dtype=np.float32, header=1)
    
    labels = label_df.loc[:, class_label]
    
    train, _ = train_test_split(labels, random_state=42)
    train = train.index 
    
    train_labeldf = label_df.loc[train, :].reset_index(drop=True)

    train = GeneExpressionData(
            datafile,
            labelfile,
            index_col=index_col,
            class_label=class_label,
            indices=train,
        )
    
    train_similar = []
    for i in range(n):
        trainsample = train[i][0]
        df_idx = train_labeldf.loc[i, index_col]
        
        trainsample_fromdf = torch.from_numpy(
            data_df.loc[df_idx, :].values
        )
        
        train_similar.append(
            all(torch.isclose(trainsample, trainsample_fromdf))
        )
        
    assert all(p for p in train_similar) # all train are similar 

def test_datamodule():
    N = 50
    datafiles, labelfiles = list(INTERIM_DATA_AND_LABEL_FILES_LIST.keys()), list(INTERIM_DATA_AND_LABEL_FILES_LIST.values())
    datafiles = [os.path.join('..', 'data', 'interim', f) for f in datafiles]
    processed_labels = [os.path.join('..', 'data', 'processed/labels', f) for f in labelfiles]

    for datafile, labelfile in zip(datafiles, processed_labels):
        print(f'Testing {datafile=}')
        _test_first_n_samples(N, datafile, labelfile)

def test_retina_data():
    N = 50 
    datafile = os.path.join('..', 'data', 'retina', 'retina_T.csv')
    labelfile = os.path.join('..', 'data', 'retina', 'retina_labels_numeric.csv')

    _test_first_n_samples(N, datafile, labelfile, 'class_label', 'cell')
    test_split(N, datafile, labelfile, 'class_label', 'cell')

if __name__ == "__main__":
    map_cols_test()
    test_datamodule()
    test_retina_data()
