import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import transformers
import mlflow

import logging
import yaml
import dotenv
from pathlib import Path

from .model_selection import load_base, load_best_logged
from .preprocessing import training_data_pipeline
from ..utils.training import train_loop, eval
from ..utils.data import encoder
from ..utils.visualization import plot_cm
from ..utils.config import Config

logger = logging.getLogger(__name__)

if __name__ == '__main__':

    dotenv.load_dotenv()
    transformers.logging.set_verbosity_error()
    logger = logging.getLogger(__name__)

    with open("./ml/main/config.yaml", "r") as f:
        config = Config(yaml.safe_load(f))

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    current_experiment = mlflow.set_experiment(config.mlflow.experiment)

    with mlflow.start_run() as run:
        device = None
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}") #
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")
        
        # load model
        load = None
        # force_base indicates whether or not to train starting from base model
        if config.model.force_base:
            load = load_base(
                checkpoint = config.model.checkpoint,
                num_labels = config.model.num_labels
            )
        # if not force_base, then look for best logged model in MLflow.
        else:
            try:
                load = load_best_logged(
                    experiment_id = current_experiment.experiment_id,
                    metrics = ['test_accuracy'],
                    ascending = [False]
                )
            # If no models logged, then load base model
            except:
                load = load_base(
                    checkpoint = config.model.checkpoint,
                    num_labels = config.model.num_labels
                )
        tokenizer = load['tokenizer']
        model = load['model']
        model = model.to(device)

        # training hyperparams (e.g. batch size, epochs, lr, etc.)
        mlflow.log_params(config.training.model_dump())

        dataloaders = training_data_pipeline(
            tokenizer = tokenizer, 
            train_start_date = config.dataset.train.start_date, 
            train_end_date = config.dataset.train.end_date,
            test_start_date = config.dataset.test.start_date,
            test_end_date = config.dataset.test.end_date,
            val_frac = config.dataset.train.val_frac,
            max_length = config.training.max_length,
            batch_size = config.training.batch_size
        )

        # dataset metadata (date ranges)
        mlflow.log_params({
            'train_start_date': config.dataset.train.start_date,
            'train_end_date': config.dataset.train.end_date,
            'val_frac': config.dataset.train.val_frac,
            'test_start_date': config.dataset.test.start_date,
            'test_end_date': config.dataset.test.end_date,
            'train_size': len(dataloaders['train'].dataset),
            'val_size': len(dataloaders['val'].dataset),
            'test_size': len(dataloaders['test'].dataset),
        })

        optimizer = torch.optim.AdamW(
            params = model.parameters(), 
            lr = config.training.learning_rate, 
            weight_decay = config.training.weight_decay
        )
        history = train_loop(
            model = model,
            optimizer = optimizer,
            train_dataloader = dataloaders['train'],
            val_dataloader = dataloaders['val'],
            epochs = config.training.epochs,
            device = device
        )
        test_metrics = eval(
            model = model,
            test_dataloader = dataloaders['test'],
            device = device
        )

        # end-of-run metrics
        mlflow.log_metrics(
            {f"final_{key}": value 
                for key, value in history['train'][-1].items() 
                if key != 'confusion_matrix'}
        )
        mlflow.log_metrics(
            {f"final_{key}": value 
                for key, value in history['val'][-1].items() 
                if key != 'confusion_matrix'}
        )
        mlflow.log_metrics(
            {f"final_{key}": value 
                for key, value in test_metrics.items() 
                if key != 'confusion_matrix'}
        )


        # log artifacts
        labels = encoder.classes_
        train_cm = plot_cm(cm = history['train'][-1]['confusion_matrix'], 
                            labels = labels, normalize = 'true',
                            annot_kws = {'size': 7})
        val_cm = plot_cm(cm = history['val'][-1]['confusion_matrix'], 
                            labels = labels, normalize = 'true',
                            annot_kws = {'size': 7})
        test_cm = plot_cm(cm = test_metrics['confusion_matrix'],
                            labels = labels, normalize = 'true',
                            annot_kws = {'size': 7})
        mlflow.log_figure(train_cm, "train_cm.png")
        mlflow.log_figure(val_cm, "val_cm.png")
        mlflow.log_figure(test_cm, "test_cm.png")

        # if first run, then log model
        if 'metrics' not in load:
            mlflow.transformers.log_model(
                transformers_model = {"model": model, "tokenizer": tokenizer},
                name = "model",
                task = "text-classification",
                pip_requirements = ['torch', 'transformers']
            )
        # otherwise, check if current model is better than best logged model
        else:
            current_test_acc = test_metrics.get('accuracy', 0.) 
            best_test_acc = load['metrics'].get('test_accuracy', 0.)
            if current_test_acc >= best_test_acc:
                mlflow.transformers.log_model(
                    transformers_model = {"model": model, "tokenizer": tokenizer},
                    name = "model",
                    task = "text-classification",
                    pip_requirements = ['torch', 'transformers']
                )
