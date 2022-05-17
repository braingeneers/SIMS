import plotly.express as px 
import plotly.graph_objects as go
import numpy as np
import scipy as sp
import pandas as pd 
import anndata as an
import matplotlib.pyplot as plt 

import sys 
sys.path.append('../src')

from model import *
from lightning_train import *
from data import *

cols = an.read_h5ad('../data/pancreas/pancreas.h5ad').var.index

plot_feature_importances('pancreas.npy', cols, 'Mouse Model, Feature Importances')
plot_explain_matrix('pancreas.npy', cols, 'Mouse Model, Explain Matrix')
