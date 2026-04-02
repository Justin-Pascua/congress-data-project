import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import Literal, List
import os
from pathlib import Path

def normalize_cm(cm: np.ndarray, mode: Literal['true', 'pred', 'all']):
    """
    Normalizes a confusion matrix.
    Args:
        cm: an unnormalized confusion matrix, where `cm[i,j]` represents the number of samples with true class i predicted as class j
        mode: a string, either 'true', 'pred', or 'all'. \\
        If 'true', then confusion matrix is normalized along rows. \\
        If 'pred', then confusion matrix is normalized along columns. \\
        If 'all', then confusion matrix is normalized over the entire matrix.
    """
    match mode:
        case 'true': denom = cm.sum(axis = 1, keepdims = True)
        case 'pred': denom = cm.sum(axis = 0, keepdims = True)
        case 'all':  denom = cm.sum()
        case _: raise ValueError(f"normalize must be 'true', 'pred', 'all', or None. Got '{normalize}'")
        
    return cm / np.maximum(denom, 1)

def plot_cm(cm: np.ndarray, labels: List[str], normalize: Literal['true', 'pred', 'all'] = None, 
            figsize: tuple = (8, 8), fontsize: int = 9,
            **kwargs):
    """
    Plots a confusion matrix using the provided labels.
    Args:
        cm: the confusion matrix to be plotted
        labels: a list of string srepresenting the class labels
        normalize: an optional string indicating how to normalize the confusion matrix. If `None`, then the matrix is left as is
        figsize: the size of the figure
        fontsize: the size of the xticklabels, yticklabels, and annotations within the cells of the matrix
    """
    if normalize is not None:
        cm = normalize_cm(cm, normalize)

    cm = np.round(cm, 3)
    
    # Create figure and explicitly set size
    plt.figure(figsize = figsize)
    
    sns.heatmap(
        cm,
        annot=True,
        xticklabels=labels,
        yticklabels=labels,
        vmin = 0,
        vmax = 1,
        annot_kws = {'size': fontsize},
        **kwargs
    )
    plt.title('Confusion Matrix', fontsize = 16)
    plt.xlabel('Pred', fontsize = 11)
    plt.ylabel('True', fontsize = 11)
    plt.xticks(fontsize = fontsize, rotation = -90, ha = 'right')
    plt.yticks(fontsize = fontsize)
    plt.tight_layout()

    return plt.gcf()

def ensure_local_image_dir(experiment, run):
    """
    Ensures that the local directory for saving images exists. Returns the path 
    to the local directory. The directory is organized by experiment and run id, i.e. 
    `images/{experiment_name}/{run_id}/`. This is used when MLflow logging of figures 
    is turned off, so that figures can still be saved locally in an organized manner.
    Args:
        experiment: the MLflow experiment object
        run: the MLflow run object
    """
    local_dir = Path("images") / experiment.name / run.info.run_id
    os.makedirs(local_dir, exist_ok = True)
    return local_dir