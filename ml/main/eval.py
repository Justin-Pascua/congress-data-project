import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import transformers
import mlflow

import logging
import yaml
import dotenv
from pathlib import Path

from .model_selection import load_model, ModelSource
from .preprocessing import eval_data_pipeline
from ..utils.train_eval import eval
from ..utils.data import raw_encoder, simplified_encoder
from ..utils.visualization import plot_cm
from ..utils.config import EvalConfig

if __name__ == '__main__':
    dotenv.load_dotenv()
    transformers.logging.set_verbosity_error()
    logger = logging.getLogger(__name__)

    with open("./ml/main/eval-config.yaml", "r") as f:
        config = EvalConfig(yaml.safe_load(f))

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    current_experiment = mlflow.set_experiment(config.mlflow.experiment)

    with mlflow.start_run(
        tags = {'mode': 'eval',
                "labels-simplified": f"{config.mlflow.labels_simplified}"}
    ) as run:
        device = None
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}") #
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")
        
        # load model
        load = load_model(
            experiment_id = current_experiment.experiment_id,
            model_id = config.model.model_id,
            checkpoint = config.model.checkpoint,
            num_labels = config.model.num_labels
        )
        tokenizer = load.tokenizer
        model = load.model
        model = model.to(device)
        if load.source == ModelSource.BASE:
            mlflow.set_tag("source", "base")
        else:
            mlflow.set_tag("source", load.model_id)

        # get test dataset
        test_dataloader = eval_data_pipeline(
            tokenizer = tokenizer,
            simplify = config.mlflow.labels_simplified,
            test_start_date = config.dataset.test.start_date,
            test_end_date = config.dataset.test.end_date,
            batch_size = 16
        )

        # log dataset metadata (date ranges)
        mlflow.log_params({
            'test_start_date': config.dataset.test.start_date,
            'test_end_date': config.dataset.test.end_date,
            'test_size': len(test_dataloader.dataset),
        })

        # eval
        test_metrics = eval(
            model = model,
            test_dataloader = test_dataloader,
            device = device
        )
        mlflow.log_metrics(
            {f"final_{key}": value 
                for key, value in test_metrics.items() 
                if key != 'confusion_matrix'}
        )

        # generate figure
        labels = (simplified_encoder.classes_ if config.mlflow.labels_simplified
                  else raw_encoder.classes_)
        figsize = (8, 8) if config.mlflow.labels_simplified else (18, 18)
        fontsize = 9 if config.mlflow.labels_simplified else 6
        test_cm = plot_cm(
            cm = test_metrics['confusion_matrix'],
            labels = labels, 
            normalize = 'true',
            figsize = figsize,
            fontsize = fontsize
        )
        mlflow.log_figure(test_cm, "test_cm.png")
