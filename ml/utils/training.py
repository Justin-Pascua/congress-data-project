import torch
from torch.utils.data import DataLoader

import sys
import time
from datetime import timedelta
import logging
from tqdm import tqdm

from .metrics import MetricAccumulator

logger = logging.getLogger(__name__)

def train_step(model, optimizer, 
               train_dataloader: DataLoader, 
               device: torch.device = torch.device("cpu"),
               return_cm: bool = False) -> dict:
    """
    Runs a single training epoch over the provided dataloader, and returns metrics.
    Args:
        model: the model to be trained.
        optimizer: the optimizer used to update model weights.
        train_dataloader: a `DataLoader` providing training batches.
        device: a `torch.device` specifying which device to run on.
        return_cm: a `bool` indicating whether to return the confusion matrix. 
    """
    model.train()

    loss = 0.
    metric_accumulator = MetricAccumulator(num_classes = 8)
    num_batches = len(train_dataloader)
    
    pbar = tqdm(total = num_batches, 
                desc = 'Train Batch', 
                bar_format = '{desc:<12}: {n_fmt}/{total_fmt} {bar:20} {elapsed} {rate_fmt}{postfix}',
                file = sys.stdout,
                leave = True)
    
    for batch in train_dataloader:
        batch = batch.to(device)
        out = model(**batch)
        optimizer.zero_grad()
        out.loss.backward()
        optimizer.step()

        loss += out.loss.item()

        true = batch['labels'].cpu()
        pred = out.logits.argmax(dim = 1).cpu()
        metric_accumulator.update(true, pred)
        pbar.update(1)
        
    full_metrics = {'loss': loss/num_batches} | metric_accumulator.compute()
    formatted_metrics = {key: f"{value:.3e}" for key, value in full_metrics.items()}
    pbar.set_postfix(formatted_metrics)
    pbar.close()

    if return_cm:
        full_metrics = full_metrics | {'confusion_matrix': metric_accumulator.get_confusion_matrix()}

    return full_metrics

def eval_step(model, 
              dataloader: DataLoader,
              device = torch.device("cpu"),
              return_cm: bool = False) -> dict:
    """
    Runs a single evaluation epoch over the provided dataloader.

    Args:
        model: The model to be evaluated.
        dataloader: a `Dataloader` providing validation batches.
        device: a `torch.device` specifying which device to run on.
        return_cm: a `bool` indicating whether to return the confusion matrix. 
    """
    model.eval()

    loss = 0.
    metric_accumulator = MetricAccumulator(num_classes = 8)
    num_batches = len(dataloader)
    
    pbar = tqdm(total = num_batches, 
                desc = 'Val Batch', 
                bar_format = '{desc:<12}: {n_fmt}/{total_fmt} {bar:20} {elapsed} {rate_fmt}{postfix}',
                file = sys.stdout,
                leave = True)
    
    for batch in dataloader:
        with torch.no_grad():
            batch = batch.to(device)
            out = model(**batch)

            loss += out.loss.item()
            true = batch['labels'].cpu()
            pred = out.logits.argmax(dim = 1).cpu()
            metric_accumulator.update(true, pred)
        pbar.update(1)

    full_metrics = {'loss': loss/num_batches} | metric_accumulator.compute()
    formatted_metrics = {key: f"{value:.3e}" for key, value in full_metrics.items()}
    pbar.set_postfix(formatted_metrics)
    pbar.close()

    if return_cm:
        full_metrics = full_metrics | {'confusion_matrix': metric_accumulator.get_confusion_matrix()}

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
        return_cm = False
        if epoch == epochs - 1:
            # only get confusion matrix on last epoch
            return_cm = True

        intermed_start = time.perf_counter()
        logger.info(f"Epoch {epoch + 1} starting")
        
        train_info = train_step(model, optimizer, train_dataloader, device, return_cm)        
        history['train'].append(train_info)
        val_info = eval_step(model, val_dataloader, device, return_cm)
        history['val'].append(val_info)
        intermed_end = time.perf_counter()
        logger.info(f"Epoch {epoch + 1} finished ({timedelta(seconds = int(intermed_end - intermed_start))})")

    end = time.perf_counter()
    logger.info(f"Training loop finished ({timedelta(seconds = int(end - start))})")
    return history
