import torch
from torch.utils.data import DataLoader
from .metrics import MetricAccumulator

def train_step(model, optimizer, 
               train_dataloader: DataLoader, 
               metric_accumulator: MetricAccumulator,
               log_every_n_steps: int = 100, 
               device = torch.device("cpu")) -> dict:
    model.train()

    loss = 0.
    metric_accumulator.reset()
    num_batches = len(train_dataloader)
    for i, batch in enumerate(train_dataloader):
        if (i+1) % log_every_n_steps == 0:
            print(f'Batch {i+1}/{num_batches}')
        batch = batch.to(device)
        out = model(**batch)
        optimizer.zero_grad()
        out.loss.backward()
        optimizer.step()

        loss += out.loss.item()

        true = batch['labels'].cpu()
        pred = out.logits.argmax(dim = 1).cpu()
        metric_accumulator.update(true, pred)

    return {'loss': loss/num_batches,
            'metrics': metric_accumulator}

def eval_step(model, 
              dataloader: DataLoader, 
              metric_accumulator: MetricAccumulator,
              log_every_n_steps: int = 100, 
              device = torch.device("cpu")) -> dict:
    model.eval()

    loss = 0.
    metric_accumulator.reset()
    num_batches = len(dataloader)
    for i, batch in enumerate(dataloader):
        if (i+1) % log_every_n_steps == 0:
            print(f'Batch {i+1}/{num_batches}')
        with torch.no_grad():
            batch = batch.to(device)
            out = model(**batch)

            loss += out.loss.item()
            true = batch['labels'].cpu()
            pred = out.logits.argmax(dim = 1).cpu()
            metric_accumulator.update(true, pred)
            

    return {'loss': loss/len(dataloader.dataset),
            'metrics': metric_accumulator}

