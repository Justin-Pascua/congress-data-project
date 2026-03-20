import numpy as np
from typing import Literal, List, Optional

class MetricAccumulator:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes

        # confusion_matrix[i, j] = num samples of true class i predicted as class j
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype = np.int64)

    def update(self, true: np.ndarray, pred: np.ndarray) -> None:
        """
        Updates confusion matrix given two numpy arrays `true` and `pred` representing true and predicted labels.
        Args:
            true: a (N,) numpy array representing true labels
            pred: a (N,) numpy array representing predicted labels
        """
        indices = true * self.num_classes + pred
        self.confusion_matrix += np.bincount(
            indices, 
            minlength = self.num_classes**2
        ).reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict:
        """
        Computes metrics (precision, recall, f1, accuracy) given current state of confusion matrix.
        Returns a dict with keys 'accuracy', 'f1', 'precision', 'recall'
        """
        cm = self.confusion_matrix.astype(np.float64)
        tp = np.diag(cm)
        fp = cm.sum(axis = 0) - tp
        fn = cm.sum(axis = 1) - tp

        precision = tp / np.maximum(tp + fp, 1)
        recall    = tp / np.maximum(tp + fn, 1)
        f1        = 2 * precision * recall / np.maximum(precision + recall, 1)
        support   = cm.sum(axis = 1)

        accuracy    = tp.sum() / cm.sum()
        w_precision = (precision * support).sum() / support.sum()
        w_recall    = (recall * support).sum() / support.sum()
        w_f1        = (f1 * support).sum() / support.sum()

        return {
            'accuracy':  accuracy,
            'f1':        w_f1,
            'precision': w_precision,
            'recall':    w_recall,
        }

    def reset(self) -> None:
        """
        Resets the state of the confusion matrix.
        """
        self.confusion_matrix.fill(0)

    def get_confusion_matrix(self, normalize: Literal['true', 'pred', 'all'] = None) -> np.ndarray:
        """
        Returns the confusion matrix, optionally normalized.
        Args:
            normalize: a string, either 'true', 'pred', or 'all'. \\
            If None, then no normalization is applied.
            If 'true', then confusion matrix is normalized along rows.
            If 'pred', then confusion matrix is normalized along columns.
            If 'all', then confusion matrix is normalized over the entire matrix.
        """
        cm = self.confusion_matrix.astype(np.float64)
    
        if normalize is None:
            return self.confusion_matrix.copy()
        
        match normalize:
            case 'true': denom = cm.sum(axis = 1, keepdims = True)
            case 'pred': denom = cm.sum(axis = 0, keepdims = True)
            case 'all':  denom = cm.sum()
            case _: raise ValueError(f"normalize must be 'true', 'pred', 'all', or None. Got '{normalize}'")
        
        return cm / np.maximum(denom, 1)

  