import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import transformers
import mlflow

import logging
import yaml
import dotenv
import os
from pathlib import Path

from .model_selection import load_model, ModelSource
from .preprocessing import eval_data_pipeline
from ..utils.train_eval import eval
from ..utils.data import raw_encoder, simplified_encoder
from ..utils.visualization import plot_cm, ensure_local_image_dir
from ..utils.config import EvalConfig

logger = logging.getLogger(__name__)

def eval_main(config: EvalConfig):
    """
    Evaluates a model on the test set.
    Args:
        config: an `EvalConfig` object specifying containing config details 
    """
    dotenv.load_dotenv()
    transformers.logging.set_verbosity_error()

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    current_experiment = mlflow.set_experiment(config.mlflow.experiment)

    with mlflow.start_run(
        tags = {'mode': 'eval'},
        description = config.mlflow.description
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
            eval_mode = True,
            experiment_id = current_experiment.experiment_id,
            model_id = config.model.model_id
        )
        tokenizer = load.tokenizer
        model = load.model
        model = model.to(device)
        logger.info("Model loaded")
        
        mlflow.set_tag("source", load.model_id)

        simplified = model.num_labels < 10
        if simplified:
            mlflow.set_tag("labels-simplified", f"{simplified}")
        else:
            mlflow.set_tag("labels-simplified", f"{simplified}")
        

        # get test dataset
        test_dataloader = eval_data_pipeline(
            tokenizer = tokenizer,
            simplify = simplified,
            test_start_date = config.dataset.test.start_date,
            test_end_date = config.dataset.test.end_date,
            batch_size = 16
        )
        logger.info("Dataset prepared")

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
        logger.info("Model evaluated")
        metrics_formatted = {f"final_{key}": value 
                             for key, value in test_metrics.items() 
                             if key != 'confusion_matrix'}
        mlflow.log_metrics(metrics_formatted)

        # generate figure
        labels = (simplified_encoder.classes_ if simplified
                  else raw_encoder.classes_)
        figsize = (8, 8) if simplified else (18, 18)
        fontsize = 9 if simplified else 6
        test_cm = plot_cm(
            cm = test_metrics['confusion_matrix'],
            labels = labels, 
            normalize = 'true',
            figsize = figsize,
            fontsize = fontsize
        )
        logger.info("Figure generated")
        if config.mlflow.log_figs:
            mlflow.log_figure(test_cm, "test_cm.png")
        else:
            local_dir = ensure_local_image_dir(current_experiment, run)
            test_cm.savefig(f"{local_dir}/test_cm.png")
            logger.info("Figure saved locally but not logged to MLflow")

        return run.info.run_id

if __name__ == '__main__':
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [
            logging.StreamHandler()
        ]
    )
    logger.info('Starting training script')
    logging.getLogger('httpx').setLevel(logging.WARNING)
    
    with open("./ml/main/eval-config.yaml", "r") as f:
        config = EvalConfig(yaml.safe_load(f))

    run_id = eval_main(config)
    print(run_id)