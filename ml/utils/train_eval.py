import torch
from torch.utils.data import DataLoader
import numpy as np
import mlflow

from collections import defaultdict
import sys
import time
from datetime import timedelta
import logging

from .metrics import MetricAccumulator

logger = logging.getLogger(__name__)

def train_step(model, optimizer, 
               train_dataloader: DataLoader, 
               device: torch.device = torch.device("cpu"),
               step_offset: int = 0,
               log_every_n_steps: int = 5) -> dict:
    """
    Runs a single training epoch over the provided dataloader, and returns metrics.
    Args:
        model: the model to be trained.
        optimizer: the optimizer used to update model weights.
        train_dataloader: a `DataLoader` providing training batches.
        device: a `torch.device` specifying which device to run on.
        step_offset: an `int` added to each batch index when logging to MLflow, used to produce globally unique step values across epochs.
        log_every_n_steps: an `int` specifying how often to log batch-based metrics to logger.
    """
    model.train()

    loss = 0.
    run_metrics = MetricAccumulator(num_classes = model.num_labels, metric_prefix = 'train')
    batch_metrics = MetricAccumulator(num_classes = model.num_labels, metric_prefix = 'train')
    num_batches = len(train_dataloader)
    
    prev_time = time.perf_counter()
    for i, batch in enumerate(train_dataloader):
        if (i+1) % log_every_n_steps == 0 or (i+1) == num_batches:
            current_time = time.perf_counter()
            logger.info(f"train batch {i + 1}/{num_batches} ({(current_time - prev_time)/log_every_n_steps:.2f}s/it)")
            prev_time = current_time

        batch = batch.to(device)
        out = model(**batch)
        optimizer.zero_grad()
        out.loss.backward()
        optimizer.step()

        loss += out.loss.item()

        true = batch['labels'].cpu()
        pred = out.logits.argmax(dim = 1).cpu()
        run_metrics.update(true, pred)
        
        # log batch-specific metrics
        batch_metrics.update(true, pred)
        mlflow.log_metrics(
            {'train_loss': out.loss.item()} | batch_metrics.compute(),
            step = step_offset + i
        )
        batch_metrics.reset()

    full_metrics = run_metrics.compute() | {'train_loss': loss/num_batches,
                                            'confusion_matrix': run_metrics.get_confusion_matrix()}

    return full_metrics

def eval_step(model, 
              dataloader: DataLoader,
              device: torch.device = torch.device("cpu"),
              metric_prefix: str = 'val',
              step_offset: int = 0,
              log_to_mlflow: bool = True,
              log_every_n_steps: int = 5) -> dict:
    """
    Runs a single evaluation epoch over the provided dataloader. Used for evaluation on validation set.

    Args:
        model: The model to be evaluated.
        dataloader: a `Dataloader` providing validation batches.
        device: a `torch.device` specifying which device to run on.
        metric_prefix: prefix applied to metric names when logging to MLflow
        step_offset: an `int` added to each batch index when logging to MLflow, used to produce globally unique step values across epochs.
        log_to_mlflow: a `bool` indicating whether to log batch-based metrics to mlflow
        log_every_n_steps: an `int` specifying how often to log batch-based metrics to logger.
    """
    model.eval()

    loss = 0.
    run_metrics = MetricAccumulator(num_classes = model.num_labels, metric_prefix = metric_prefix)
    batch_metrics = MetricAccumulator(num_classes = model.num_labels, metric_prefix = metric_prefix)
    num_batches = len(dataloader)
    
    prev_time = time.perf_counter()
    for i, batch in enumerate(dataloader):
        if (i+1) % log_every_n_steps == 0 or (i+1) == num_batches:
            current_time = time.perf_counter()
            logger.info(f"{metric_prefix} batch {i + 1}/{num_batches} ({(current_time - prev_time)/log_every_n_steps:.2f}s/it)")
            prev_time = current_time

        with torch.no_grad():
            batch = batch.to(device)
            out = model(**batch)

            loss += out.loss.item()

            true = batch['labels'].cpu()
            pred = out.logits.argmax(dim = 1).cpu()
            run_metrics.update(true, pred)
            
            # log batch-specific metrics
            if log_to_mlflow:
                batch_metrics.update(true, pred)
                mlflow.log_metrics(
                    {f'{metric_prefix}_loss': out.loss.item()} | batch_metrics.compute(),
                    step = step_offset + i
                )
                batch_metrics.reset()

    # metric prefix needed for separating train and val metrics in mlflow,
    # but not needed for confusion matrix because we don't log the cm directly. 
    # instead we log a figure generated from the cm
    full_metrics = run_metrics.compute()| {f'{metric_prefix}_loss': loss/num_batches,
                                           'confusion_matrix': run_metrics.get_confusion_matrix()}

    return full_metrics

def train_loop(model, optimizer,
               train_dataloader: DataLoader,
               val_dataloader: DataLoader,
               epochs: int,
               device = torch.device("cpu")
               ) -> list:
    """
    Runs the full training loop over a specified number of epochs.
    Args:
        model: the model to be trained.
        optimizer: the optimizer used to update model weights.
        train_dataloader: a `DataLoader` providing training batches.
        val_dataloader: a `Dataloader` providing validation batches.
        epochs: an `int` specifying the number of epochs to train for.
        device: a `torch.device` specifying which device to run on.
    """
    start = time.perf_counter()
    logger.info(f"Beginning training loop")

    history = {'train': [], 'val': []}    
    
    for epoch in range(epochs):
        intermed_start = time.perf_counter()
        logger.info(f"Epoch {epoch + 1} starting")
        
        train_info = train_step(
            model = model, 
            optimizer = optimizer, 
            train_dataloader = train_dataloader, 
            device = device,
            step_offset = epoch * len(train_dataloader)
        )        
        history['train'].append(train_info)
        
        val_info = eval_step(
            model = model, 
            dataloader = val_dataloader, 
            device = device,
            step_offset = epoch * len(val_dataloader)
        )
        history['val'].append(val_info)
        
        intermed_end = time.perf_counter()
        logger.info(f"Epoch {epoch + 1} finished ({timedelta(seconds = int(intermed_end - intermed_start))})")

    end = time.perf_counter()
    logger.info(f"Training loop finished ({timedelta(seconds = int(end - start))})")
    return history

def eval(model, 
         test_dataloader: DataLoader,
         device: torch.device = torch.device("cpu"),
         ) -> dict:
    """
    Computes evaluation metrics on a test dataset.
    Args:
        model: The model to be evaluated.
        test_dataloader: a `Dataloader` providing validation batches.
        device: a `torch.device` specifying which device to run on.
    """
    metrics = eval_step(model, test_dataloader, device, metric_prefix = 'test', log_to_mlflow = False)
    return metrics

def inference_eval(model,
                   test_dataloader: DataLoader,
                   device: torch.device = torch.device("cpu"),
                   log_every_n_steps: int = 10
                   ) -> dict:
    """
    Used for inference time evaluation, where we chunk each sample into multiple pieces 
    and want to aggregate predictions across chunks before computing metrics. 
    Args:
        model: The model to be evaluated.
        test_dataloader: a `Dataloader` providing validation batches.
        device: a `torch.device` specifying which device to run on.
        log_every_n_steps: an `int` specifying how often to log batch-based metrics to logger.
    """
    start_time = time.perf_counter()
    logger.info("Beginning inference evaluation")
    doc_logits = defaultdict(lambda: torch.tensor([0.0] * model.num_labels))
    doc_targets = dict()

    prev_time = time.perf_counter()
    with torch.no_grad():
        for batch_num, batch in enumerate(test_dataloader):
            if (batch_num + 1) % log_every_n_steps == 0 or (batch_num + 1) == len(test_dataloader):
                current_time = time.perf_counter()
                logger.info(f"test batch {batch_num + 1}/{len(test_dataloader)} "
                            f"({(current_time - prev_time)/log_every_n_steps:.2f}s/it)")
                prev_time = current_time

            tokenized = {'input_ids': batch['input_ids'].to(device),
                        'attention_mask': batch['attention_mask'].to(device)}
            parent_idxs = batch['parent_indices'].to(device)
            targets = batch['labels'].to(device)

            logits = model(**tokenized).logits

            for group_idx, logit, target in zip(parent_idxs, logits, targets):
                doc_logits[group_idx.item()] += logit.cpu()
                doc_targets[group_idx.item()] = target.item()

        sorted_idxs = sorted(doc_logits.keys())
        preds = np.array([doc_logits[i].argmax().item() for i in sorted_idxs])
        targets = np.array([doc_targets[i] for i in sorted_idxs])

    metric_accum = MetricAccumulator(num_classes = model.num_labels, metric_prefix = 'test')
    metric_accum.update(targets, preds)
    metrics = metric_accum.compute() | {'confusion_matrix': metric_accum.get_confusion_matrix()}
    
    end_time = time.perf_counter()
    logger.info(f"Inference evaluation complete ({end_time - start_time:.2f}s)")
    
    return metrics
