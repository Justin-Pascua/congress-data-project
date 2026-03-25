import mlflow
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List

def load_best_logged(experiment_id: str, metrics: List[dict], ascending: List[bool]):    
    sorted_models = mlflow.search_logged_models(
        experiment_ids = [experiment_id],
        order_by = [{"field_name": f"metrics.{metric}", 
                     "ascending": asc} for metric, asc in zip(metrics, ascending)]
    )
    best_model = sorted_models.iloc[0]
    metrics = {metric_obj.key: metric_obj.value for metric_obj in best_model['metrics']}
    
    components = mlflow.transformers.load_model(
        model_uri = f"mlruns/{experiment_id}/models/{best_model["model_id"]}/artifacts/",
        return_type = "components"
    )
    return {'model': components['model'],
            'tokenizer': components['tokenizer'],
            'metrics': metrics}

def load_base(checkpoint: str, num_labels: int):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels = num_labels)
    return {'model': model,
            'tokenizer': tokenizer}

def compare(best_metrics, current_metrics, criteria):
    pass

