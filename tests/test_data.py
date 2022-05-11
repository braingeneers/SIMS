import unittest 
import sys
import os 
import random 
import pandas as pd 
import anndata as an 
import pandas as pd 
import numpy as np
import torch
import pathlib 
from torch.utils.data import *

random.seed(42)
from os.path import join, dirname, abspath 
sys.path.append(join(dirname(abspath(__file__)), '..', 'src'))

from models.lib.data import *
from helper import *

class TestData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.here = pathlib.Path(__file__).parent.resolve()

        cls.datapath = os.path.join(cls.here, 'datapath')
        if not os.path.isdir(cls.datapath):
            print(f'Making test directories')
            os.makedirs(cls.datapath, exist_ok=True)

        # Define test file locations 
        cls.datafile_csv = os.path.join(cls.datapath, 'test_data.csv')
        cls.datafile_h5ad = os.path.join(cls.datapath, 'test_data.h5ad')
        cls.labelfile = os.path.join(cls.datapath, 'test_labels.csv')
    
    def test_dataset_from_csv(self):
        # Create a test dataframe 
        df = pd.DataFrame(index=range(25), columns=[f'col_{i}' for i in range(10)])

        # Such that row_i = (i, ... ,i)
        for i in range(25): 
            df.loc[i, :] = [i]*10

        # Define label dataframe
        labels = pd.DataFrame(
            index=range(25), 
            columns=['index_col', 'label']
        )

        # Create fake index_col = label col such that index_col_i = label_col_i = row_i = (i,...,i)
        labels['index_col'] = [int(x) for x in random.sample(list(df.index), k=len(df))]
        labels['label'] = labels['index_col']

        labels.to_csv(self.labelfile, index=False)
        df.to_csv(self.datafile_csv)

        # Now, write the actual tests 
        train, val, test = generate_single_dataset(
            datafile=self.datafile_csv,
            labelfile=self.labelfile,
            class_label='label',
            index_col='index_col',
            skip=2,
            stratify=False,
        )

        # Test train 
        for i in range(len(train)):
            X, y = train[i]
            self.assertEqual(X[0], y)

        # Test val 
        for i in range(len(val)):
            X, y = val[i]
            self.assertEqual(X[0], y)

        # Test test  
        for i in range(len(test)):
            X, y = test[i]
            self.assertEqual(X[0], y)
            
    def test_dataset_from_h5ad(self):
        # Create a test dataframe 
        df = pd.DataFrame(index=range(25), columns=[f'col_{i}' for i in range(10)])

        # Such that row_i = (i, ... ,i)
        for i in range(25): 
            df.loc[i, :] = [i]*10
        
        # Convert this to an anndata object 
        cols = df.columns 

        df = an.AnnData(df.values)
        df.var.index = cols

        # Define label dataframe
        labels = pd.DataFrame(
            index=range(25), 
            columns=['index_col', 'label']
        )   

        # Create fake index_col = label col such that index_col_i = label_col_i = row_i = (i,...,i)
        labels = pd.DataFrame(index=range(25), columns=['index_col', 'label'])
        labels['index_col'] = [int(x) for x in random.sample(list(df.obs.index), k=len(df))]
        labels['label'] = labels['index_col']

        labels.to_csv(self.labelfile, index=False)
        df.write_h5ad(self.datafile_h5ad)

        # Now, write the actual tests 
        # Don't stratify since we have len(rows) = card(labels)
        train, val, test = generate_single_dataset(
            datafile=self.datafile_h5ad,
            labelfile=self.labelfile,
            class_label='label',
            index_col='index_col',
            stratify=False,
        )

        # Test train 
        for i in range(len(train)):
            X, y = train[i]
            self.assertEqual(X[0], y)

        # Test val 
        for i in range(len(val)):
            X, y = val[i]
            self.assertEqual(X[0], y)

        # Test test  
        for i in range(len(test)):
            X, y = test[i]
            self.assertEqual(X[0], y)

    def test_refgene_alignment(self):
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

    def test_dataloader_from_csv(self):
        train, val, test = generate_dataloaders(
            datafiles=[self.datafile_csv],
            labelfiles=[self.labelfile],
            class_label='label',
            index_col='index_col',
            skip=2,
            stratify=False,
        )

        for sample in train:
            X, Y = sample[0], sample[1]
            for x, y in zip(X, Y):
                self.assertEqual(x[0], y)

        for sample in val:
            X, Y = sample[0], sample[1]
            for x, y in zip(X, Y):
                self.assertEqual(x[0], y)

        for sample in test:
            X, Y = sample[0], sample[1]
            for x, y in zip(X, Y):
                self.assertEqual(x[0], y)

    def test_dataloader_from_h5ad(self):
        train, val, test = generate_dataloaders(
            datafiles=[self.datafile_h5ad],
            labelfiles=[self.labelfile],
            class_label='label',
            index_col='index_col',
            stratify=False,
        )

        for sample in train:
            X, Y = sample[0], sample[1]
            for x, y in zip(X, Y):
                self.assertEqual(x[0], y)

        for sample in val:
            X, Y = sample[0], sample[1]
            for x, y in zip(X, Y):
                self.assertEqual(x[0], y)

        for sample in test:
            X, Y = sample[0], sample[1]
            for x, y in zip(X, Y):
                self.assertEqual(x[0], y)

if __name__ == "__main__":
    unittest.main()
    