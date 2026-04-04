import torch
import transformers
import mlflow

import logging
import yaml
import dotenv
import os

from .model_selection import load_model, ModelSource
from .preprocessing import training_data_pipeline, eval_data_pipeline
from ..utils.train_eval import train_loop, inference_eval
from ..utils.data import raw_encoder, simplified_encoder
from ..utils.visualization import plot_cm, ensure_local_image_dir
from ..utils.config import TrainConfig

logger = logging.getLogger(__name__)

def train_main(config: TrainConfig):
    """
    Trains model, and evaluates on a validation and test set.
    Args:
        config: a `TrainConfig` object containing config details
    """
    dotenv.load_dotenv()
    transformers.logging.set_verbosity_error()

    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
    current_experiment = mlflow.set_experiment(config.mlflow.experiment)

    with mlflow.start_run(
        tags = {'mode': 'train',
                "labels-simplified": f"{config.mlflow.labels_simplified}"},
        description = config.mlflow.description
    ) as run:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using {device}")
        
        # load model
        load = load_model(
            experiment_id = current_experiment.experiment_id,
            force_base = config.model.force_base,
            checkpoint = config.model.checkpoint,
            num_labels = config.model.num_labels
        )
        tokenizer = load.tokenizer
        model = load.model
        model = model.to(device)
        logger.info("Model loaded")

        if load.source == ModelSource.BASE:
            mlflow.set_tag("source", "base")
        else:
            mlflow.set_tag("source", load.model_id)

        # training hyperparams (e.g. batch size, epochs, lr, etc.)
        mlflow.log_params(config.training.model_dump())

        train_val_dataloaders = training_data_pipeline(
            tokenizer = tokenizer, 
            simplify = config.mlflow.labels_simplified,
            train_start_date = config.dataset.train.start_date, 
            train_end_date = config.dataset.train.end_date,
            weighted_sampling = config.dataset.train.weighted_sampling,
            val_frac = config.dataset.train.val_frac,
            max_length = config.training.max_length,
            batch_size = config.training.batch_size
        )
        test_dataloader = eval_data_pipeline(
            tokenizer = tokenizer, 
            simplify = config.mlflow.labels_simplified,
            test_start_date = config.dataset.test.start_date, 
            test_end_date = config.dataset.test.end_date,
            max_length = config.training.max_length,
            batch_size = config.training.batch_size
        )
        logger.info("Datasets prepared")

        # dataset metadata (date ranges)
        mlflow.log_params({
            'train_start_date': config.dataset.train.start_date,
            'train_end_date': config.dataset.train.end_date,
            'val_frac': config.dataset.train.val_frac,
            'weighted_sampling': config.dataset.train.weighted_sampling,
            'test_start_date': config.dataset.test.start_date,
            'test_end_date': config.dataset.test.end_date,
            'train_num_bills': train_val_dataloaders['train'].dataset.num_bills,
            'train_num_chunks': train_val_dataloaders['train'].dataset.num_chunks,
            'val_num_bills': train_val_dataloaders['val'].dataset.num_bills,
            'val_num_chunks': train_val_dataloaders['val'].dataset.num_chunks,
            'test_num_bills': test_dataloader.dataset.num_bills,
            'test_num_chunks': test_dataloader.dataset.num_chunks,
        })

        optimizer = torch.optim.AdamW(
            params = model.parameters(), 
            lr = config.training.learning_rate, 
            weight_decay = config.training.weight_decay
        )
        history = train_loop(
            model = model,
            optimizer = optimizer,
            train_dataloader = train_val_dataloaders['train'],
            val_dataloader = train_val_dataloaders['val'],
            epochs = config.training.epochs,
            device = device
        )
        test_metrics = inference_eval(
            model = model,
            test_dataloader = test_dataloader,
            device = device
        )
        logger.info("Model trained")

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

        # log plots
        labels = (simplified_encoder.classes_ if config.mlflow.labels_simplified
                  else raw_encoder.classes_)
        figsize = (8, 8) if config.mlflow.labels_simplified else (18, 18)
        fontsize = 9 if config.mlflow.labels_simplified else 6
        train_cm = plot_cm(
            cm = history['train'][-1]['confusion_matrix'], 
            labels = labels, 
            normalize = 'true',
            figsize = figsize,
            fontsize = fontsize
        )
        val_cm = plot_cm(
            cm = history['val'][-1]['confusion_matrix'], 
            labels = labels, 
            normalize = 'true',
            figsize = figsize,
            fontsize = fontsize
        )
        test_cm = plot_cm(
            cm = test_metrics['confusion_matrix'],
            labels = labels, 
            normalize = 'true',
            figsize = figsize,
            fontsize = fontsize
        )
        logger.info("Figures generated")
        if config.mlflow.log_figs:
            mlflow.log_figure(train_cm, "train_cm.png")
            mlflow.log_figure(val_cm, "val_cm.png")
            mlflow.log_figure(test_cm, "test_cm.png")
        else:
            local_dir = ensure_local_image_dir(current_experiment, run)
            train_cm.savefig(f"{local_dir}/train_cm.png")
            val_cm.savefig(f"{local_dir}/val_cm.png")
            test_cm.savefig(f"{local_dir}/test_cm.png")
            logger.info("Figure saved locally but not logged to MLflow")


        # if training from base, then log model
        if load.source == ModelSource.BASE:
            mlflow.transformers.log_model(
                transformers_model = {"model": model, "tokenizer": tokenizer},
                name = "model",
                task = "text-classification",
                pip_requirements = ['torch', 'transformers']
            )
        # otherwise, check if current model is better than best logged model
        else:
            current_test_acc = test_metrics.get('accuracy', 0.) 
            best_test_acc = load.metrics.get('test_accuracy', 0.)
            if current_test_acc >= best_test_acc:
                mlflow.transformers.log_model(
                    transformers_model = {"model": model, "tokenizer": tokenizer},
                    name = "model",
                    task = "text-classification",
                    pip_requirements = ['torch', 'transformers']
                )

    return run.info.run_id

if __name__ == '__main__':
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt = '%Y-%m-%d %H:%M:%S',
        handlers = [
            logging.StreamHandler()
        ]
    )
    logger.info('Starting training script')
    logging.getLogger('httpx').setLevel(logging.WARNING)

    with open("./ml/main/train-config.yaml", "r") as f:
        config = TrainConfig(yaml.safe_load(f))

    run_id = train_main(config)
    print(run_id)