import mlflow
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Any, Optional
from dataclasses import dataclass, replace
from enum import Enum
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelSource(Enum):
    """
    Enum used to indicate where a model has been loaded from.
    """
    BASE = "base"
    LOGGED = "logged"

@dataclass
class ModelLoad:
    """
    Dataclass for storing loaded model, tokenizer, and metadata
    """
    model: Any
    tokenizer: Any
    source: ModelSource
    model_id: Optional[str] = None
    metrics: Optional[dict] = None

def get_model_uri(experiment_id, model_id) -> Path:
    """
    Returns path to model artifacts for a given model id and experiment id. 
    Assumes mlflow is configured to store artifacts in ./mlflow_data/mlruns
    Args:
        experiment_id: The id of the MLflow experiment to load from
        model_id: The id of the model to load
    """
    experiment_name = mlflow.get_experiment(experiment_id).name
    base = Path("./mlflow_data/mlruns") 
    model_uri = base / experiment_name / "models" / model_id / "artifacts"
    return model_uri

def load_logged(experiment_id: str, model_id: str) -> ModelLoad:
    """
    Loads model and tokenizer from MLflow given an experiment id and model id. 
    Args:
        experiment_id: The id of the MLflow experiment to load from
        model_id: The id of the model to load
    """
    model_uri = get_model_uri(experiment_id, model_id)
    components = mlflow.transformers.load_model(
        model_uri = model_uri,
        return_type = "components"
    )

    return ModelLoad(
        model = components['model'],
        tokenizer = components['tokenizer'],
        source = ModelSource.LOGGED,
        model_id = model_id
    )

def load_best_logged(experiment_id: str, metrics: List[str]) -> ModelLoad:
    """
    Loads the best model from MLflow given an experiment id and a list of 
    metrics to sort by.
    Args:
        experiment_id: The id of the MLflow experiment to load from
        metrics: A list of metric names to sort by. Models will be sorted 
            by these metrics in descending order
    """
    sorted_models = mlflow.search_logged_models(
        experiment_ids = [experiment_id],
        order_by = [{"field_name": f"metrics.{metric}", "ascending": False} 
                    for metric in metrics]
    )
    best_model = sorted_models.iloc[0]
    metrics = {metric_obj.key: metric_obj.value for metric_obj in best_model['metrics']}
    
    model_id = best_model["model_id"]

    load = load_logged(experiment_id, model_id)

    return replace(load, metrics = metrics)

def load_base(checkpoint: str, num_labels: int) -> ModelLoad:
    """
    Loads base model and tokenizer from HuggingFace given a checkpoint 
    and number of labels. 
    Args:
        checkpoint: The HuggingFace checkpoint to load from
        num_labels: The number of labels for the sequence classification head
    """
    if checkpoint is None:
        raise ValueError("Can't load base model with None as checkpoint")
    if num_labels is None:
        raise ValueError("Can't load base model with None as num_labels")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels = num_labels)
    
    return ModelLoad(
        model =  model,
        tokenizer = tokenizer,
        source = ModelSource.BASE
    )

def load_model(eval_mode: bool = False,
               experiment_id: str = None,
               model_id: str = None,
               force_base: bool = False, 
               checkpoint: str = None, num_labels: int = None) -> ModelLoad:
    """
    Loads model and tokenizer based on provided arguments. If force_base is True,
    then loads base model from HuggingFace using checkpoint and num_labels. 
    Otherwise, if model_id is provided, loads the corresponding model from MLflow.
    If neither force_base nor model_id is provided, then looks for the best model
    in MLflow and loads it. If no models are found in MLflow, then loads base model 
    from HuggingFace. 
    Args:
        eval_mode: Whether or not the model will be loaded for evaluation. If True, 
            then force_base cannot be True, since base model is only used for training.
        experiment_id: The id of the MLflow experiment to load from. Required if 
            model_id is provided
        model_id: The id of the model to load from MLflow. If not provided, 
            then looks for best model in MLflow
        force_base: Whether or not to force loading the base model from HuggingFace. 
            If True, then checkpoint and num_labels must be provided, and model_id is ignored
        checkpoint: The HuggingFace checkpoint to load from if force_base is True
        num_labels: The number of labels for the sequence classification head

    """
    # if need to load base model
    if force_base:
        # then check that checkpoint and num_labels is provided
        if checkpoint is None or num_labels is None:
            raise ValueError(f"Must specify 'checkpoint' and 'num_labels' in order to load base model")

    # load model
    load = None
    # force_base indicates whether or not to train starting from base model
    if force_base:
        if eval_mode:
            raise ValueError("Tried to load base model in eval mode")
        logger.info("force_base is True, loading base model")
        load = load_base(
            checkpoint = checkpoint,
            num_labels = num_labels
        )
    elif model_id is not None:
        logger.info("Model id provided, loading from MLflow")
        load = load_logged(
            experiment_id = experiment_id,
            model_id = model_id
        )
    # if not force_base, then look for best logged model in MLflow.
    else:
        try:
            logger.info("No model id provided, loading best from MLflow")
            load = load_best_logged(
                experiment_id = experiment_id,
                metrics = ['final_test_f1']
            )
        # If no models logged, then load base model
        except:
            logger.info("No models found in MLflow, loading base")
            if eval_mode:
                raise ValueError("Tried to load base model in eval mode")
            load = load_base(
                checkpoint = checkpoint,
                num_labels = num_labels
            )
    return load
