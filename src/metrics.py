from typing import *

import torch 
import numpy as np 
from torchmetrics.functional import *
import torchmetrics 
from tqdm import tqdm
import sys

sys.path.append('../src')
from data import *
from lightning_train import *
from model import *
from functools import partial 

def confusion_matrix(model, dataloader, num_classes):
    confusion_matrix = torch.zeros(num_classes, num_classes)
    with torch.no_grad():
        for i, (inputs, classes) in enumerate(tqdm(dataloader)):
            outputs, _ = model(inputs)
            
            _, preds = torch.max(outputs, 1)
            for t, p in zip(classes.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
                    
    return confusion_matrix 

def median_f1(tps, fps, fns):
    precisions = tps / (tps+fps)
    recalls = tps / (tps+fns)
    
    f1s = 2*(np.dot(precisions, recalls)) / (precisions + recalls)
    
    return np.nanmedian(f1s)

def per_class_f1(*args, **kwargs):
    res = torchmetrics.functional.f1_score(*args, **kwargs, average='none')
    return res

def per_class_precision(*args, **kwargs):
    res = torchmetrics.functional.precision(*args, **kwargs, average='none')
    
    return res

def per_class_recall(*args, **kwargs):
    res = torchmetrics.functional.precision(*args, **kwargs, average='none')
    
    return res 

def weighted_accuracy(*args, **kwargs):
    res = torchmetrics.functional.accuracy(*args, **kwargs, average='weighted')
    
    return res 

def balanced_accuracy(*args, **kwargs):
    res = torchmetrics.functional.accuracy(*args, **kwargs, average='macro')
    
    return res 

def aggregate_metrics(num_classes) -> Dict[str, Callable]:
    metrics = {
        # Accuracies
        'total_accuracy': torchmetrics.functional.accuracy,
        'balanced_accuracy': partial(balanced_accuracy, num_classes=num_classes),
        'weighted_accuracy': partial(weighted_accuracy, num_classes=num_classes),
        
        # Precision, recall and f1s
        'precision': torchmetrics.functional.precision,
        'recall': torchmetrics.functional.recall,
        'f1': torchmetrics.functional.f1_score,
        
        # Per class 
        'per_class_f1': partial(per_class_f1, num_classes=num_classes),
        'per_class_precision': partial(per_class_precision, num_classes=num_classes),
        'per_class_recall': partial(per_class_recall, num_classes=num_classes),
        
        # Random stuff I might want
        'specificity': partial(torchmetrics.functional.specificity, num_classes=num_classes),
        'confusion_matrix': partial(torchmetrics.functional.confusion_matrix, num_classes=num_classes),
        'auroc': partial(torchmetrics.functional.auroc, num_classes=num_classes)
    }
    
    return metrics 