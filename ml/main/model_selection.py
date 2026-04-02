import mlflow
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Any, Optional, Literal
from dataclasses import dataclass, replace
from enum import Enum, auto
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
    model: Any
    tokenizer: Any
    source: ModelSource
    model_id: Optional[str] = None
    metrics: Optional[dict] = None

def get_model_uri(experiment_id, model_id) -> Path:
    experiment_name = mlflow.get_experiment(experiment_id).name
    base = Path("./mlflow_data/mlruns") 
    model_uri = base / experiment_name / "models" / model_id / "artifacts"
    return model_uri

def load_logged(experiment_id: str, model_id: str) -> ModelLoad:
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
