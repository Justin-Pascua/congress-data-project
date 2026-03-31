import mlflow
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Any, Optional
from dataclasses import dataclass
from enum import Enum, auto

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

def load_logged(experiment_id: str, model_id: str) -> ModelLoad:
    components = mlflow.transformers.load_model(
        model_uri = f"mlruns/{experiment_id}/models/{model_id}/artifacts/",
        return_type = "components"
    )
    return ModelLoad(
        model = components['model'],
        tokenizer = components['tokenizer'],
        source = ModelSource.LOGGED,
        model_id = model_id
    )

def load_best_logged(experiment_id: str, metrics: List[dict]) -> ModelLoad:
    sorted_models = mlflow.search_logged_models(
        experiment_ids = [experiment_id],
        order_by = [{"field_name": f"metrics.{metric}"} for metric in metrics]
    )
    best_model = sorted_models.iloc[0]
    metrics = {metric_obj.key: metric_obj.value for metric_obj in best_model['metrics']}
    
    components = mlflow.transformers.load_model(
        model_uri = f"mlruns/{experiment_id}/models/{best_model["model_id"]}/artifacts/",
        return_type = "components"
    )
    return ModelLoad(
        model =  components['model'],
        tokenizer = components['tokenizer'],
        source = ModelSource.LOGGED,
        model_id = best_model["model_id"],
        metrics = metrics
    )

def load_base(checkpoint: str, num_labels: int) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels = num_labels)
    
    return ModelLoad(
        model =  model,
        tokenizer = tokenizer,
        source = ModelSource.BASE
    )

def load_model(experiment_id: str = None,
               model_id: str = None,
               force_base: bool = False, 
               checkpoint: str = None, num_labels: int = None) -> ModelLoad:
    # if need to load base model
    if force_base or experiment_id is None:
        # then check that checkpoint and num_labels is provided
        if checkpoint is None or num_labels is None:
            raise ValueError(f"Must specify 'checkpoint' and 'num_labels' in order to load base model")

    # load model
    load = None
    # force_base indicates whether or not to train starting from base model
    if force_base:
        load = load_base(
            checkpoint = checkpoint,
            num_labels = num_labels
        )
    elif model_id is not None:
        load = load_logged(
            experiment_id = experiment_id,
            model_id = model_id
        )
    # if not force_base, then look for best logged model in MLflow.
    else:
        try:
            load = load_best_logged(
                experiment_id = experiment_id,
                metrics = ['test_accuracy']
            )
        # If no models logged, then load base model
        except:
            load = load_base(
                checkpoint = checkpoint,
                num_labels = num_labels
            )
    return load
