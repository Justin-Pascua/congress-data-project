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
        
    tokenizer, model = None, None
    try:
        tokenizer = AutoTokenizer.from_pretrained(config['model']['checkpoint']['finetuned'])
        model = AutoModelForSequenceClassification.from_pretrained(config['model']['checkpoint']['finetuned'])
    except:
        tokenizer = AutoTokenizer.from_pretrained(
            config['model']['checkpoint']['base']
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            config['model']['checkpoint']['base'], 
            num_labels = config['model']['num_labels']
        )
    model = model.to(device)

    dataloaders = training_data_pipeline(
        tokenizer = tokenizer, 
        train_start_date = config['dataset']['train']['start_date'],
        train_end_date = config['dataset']['train']['end_date'],
        test_start_date = config['dataset']['test']['start_date'],
        test_end_date = config['dataset']['test']['end_date'],
        val_frac = config['dataset']['train']['val_frac'],
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
        train_dataloader = dataloaders['train'],
        val_dataloader = dataloaders['val'],
        epochs = config['training']['epochs'],
        device = device
    )
    
    