import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import Literal, List

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

def plot_cm(cm: np.ndarray, labels: List[str], normalize: Literal['true', 'pred', 'all'] = None, **kwargs):
    """
    Plots a confusion matrix using the provided labels.
    Args:
        cm: the confusion matrix to be plotted
        labels: a list of string srepresenting the class labels
        normalize: an optional string indicating how to normalize the confusion matrix. If `None`, then the matrix is left as is.
    """
    if normalize is not None:
        cm = normalize_cm(cm, normalize)
    
    cm = np.round(cm, 3)
    fig, ax = plt.subplots(figsize = (7, 7))
        
    sns.heatmap(
        cm,
        annot = True,
        xticklabels = labels,
        yticklabels = labels,
        **kwargs
    )
    ax.set_xlabel('Pred', fontsize = 11)
    ax.set_ylabel('True', fontsize = 11)
    ax.set_title('Confusion Matrix')
    fig.tight_layout()
    plt.xticks(fontsize = 8, rotation = 45, ha = 'right')
    plt.yticks(fontsize = 8)

    return fig