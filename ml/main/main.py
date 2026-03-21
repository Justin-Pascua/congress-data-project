import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import yaml
import dotenv

from database.read import read_bills
from .preprocessing import training_data_pipeline
from ..utils.training import train_loop

if __name__ == '__main__':

    dotenv.load_dotenv()
    
    with open("./ml/main/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}") #
    else:
        device = torch.device("cpu")
        print("Using CPU")
        

    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['pretrained_checkpoint']
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        config['model']['pretrained_checkpoint'], 
        num_labels = config['model']['num_labels']
    )
    model = model.to(device)

    bills = read_bills(
        congress_num = config['dataset']['congress_num'],
        start_date = config['dataset']['start_date']
    )
    data = training_data_pipeline(
        tokenizer = tokenizer, 
        bills = bills,
        test_frac = config['dataset']['test_frac'],
        val_frac = config['dataset']['val_frac'],
        max_length = config['training']['max_length'],
        batch_size = config['training']['batch_size']
    )

    optimizer = torch.optim.AdamW(
        params = model.parameters(), 
        lr = config['training']['learning_rate'], 
        weight_decay = config['training']['weight_decay']
    )
    history = train_loop(
        model = model,
        optimizer = optimizer,
        train_dataloader = data['dataloaders']['train'],
        val_dataloader = data['dataloaders']['val'],
        epochs = config['training']['epochs'],
        device = device
    )
    
    