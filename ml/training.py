import torch
from torch.utils.data import DataLoader

def train_step(model, optimizer, train_dataloader: DataLoader, log_every_n_steps: int = 100) -> None:
    model.train()

    num_batches = len(train_dataloader)
    for i, batch in enumerate(train_dataloader):
        if (i+1) % log_every_n_steps == 0:
            print(f'Batch {i+1}/{num_batches}')
        out = model(**batch)
        optimizer.zero_grad()
        out.loss.backward()
        optimizer.step()

def eval_step(model, dataloader: DataLoader, log_every_n_steps: int = 100) -> dict:
    model.eval()
    
    loss = 0.
    correct = 0
    num_batches = len(dataloader)
    for i, batch in enumerate(dataloader):
        if (i+1) % log_every_n_steps == 0:
            print(f'Batch {i+1}/{num_batches}')
        with torch.no_grad():
            out = model(**batch)
            loss += out.loss

            true = batch['labels']
            pred = out.logits.argmax(dim = 1)
            correct = (true == pred).sum()
            
    return {'loss': loss,
            'accuracy': correct/len(dataloader.dataset)}

