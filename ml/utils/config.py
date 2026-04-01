from pydantic import BaseModel
from typing import Optional
from datetime import datetime, date

class MLflowConfigBase(BaseModel):
    """
    MLflow tracking configuration 
    """
    experiment: str
    description: Optional[str] = None   # description given to run

class TrainMLflowConfig(MLflowConfigBase):
    """
    MLflow tracking configuration for training 
    """
    labels_simplified: bool  # indicates if original classes have been grouped into broader classes

class EvalMLflowConfig(MLflowConfigBase):
    """
    MLflow tracking configuration for evaluation
    """

class TrainModelConfig(BaseModel):
    """
    Model config for training.
    """
    checkpoint: str     # HuggingFace checkpoint of base model
    num_labels: int     # number of classes in the classification tas
    force_base: bool    # If True, then loads base model. If False, then loads best model logged in MLflow (or base model if no models logged in MLflow) 
    
class EvalModelConfig(BaseModel):
    """
    Model config for evaluation.
    """
    model_id: Optional[str] = None  # MLflow run/model ID to load; defaults to best model if omitted

class TrainingConfig(BaseModel):
    """
    Hyperparameters and training loop configuration.
    """
    epochs: int
    learning_rate: float        
    weight_decay: float
    batch_size: int
    max_length: Optional[int]   # max token length of samples

class TrainDatasetConfig(BaseModel):
    """
    Date-range and split configuration for the training dataset.
    """
    start_date: datetime | date   # dates specify date range to pull samples from
    end_date: datetime | date
    val_frac: float               # specifies what fraction of data to use for validation
    weighted_sampling: bool

class TestDatasetConfig(BaseModel):
    """
    Date-range configuration for the test dataset.
    """
    start_date: datetime | date         # dates specify date range to pull samples from
    end_date: datetime | date | None    # if end date is None, then data preprocessing pipeline will treat this as datetime.now()  

class DatasetConfig(BaseModel):
    """
    Container for train and test dataset configurations.
    """
    train: Optional[TrainDatasetConfig] = None
    test: TestDatasetConfig

class TrainConfig(BaseModel):
    """
    Top-level configuration object, populated from a YAML config file via yaml.safe_load.

    Expected YAML structure::

        mlflow:
          experiment: my-experiment-name
          simplified_labels: False

        model:
          num_labels: 33
          checkpoint: "distilbert/distilroberta-base"
          force_base: True

        training:
          epochs: 100
          learning_rate: 1.0e-5
          weight_decay: 0.001
          batch_size: 16
          max_length: 512

        dataset:
          train:
            start_date: 2025-01-01
            end_date: 2025-12-31
            val_frac: 0.25
          test:
            start_date: 2026-01-01
            end_date:        # Leave blank to default to datetime.now() at runtime

    Usage::

        import yaml
        from config import Config

        with open("config.yaml") as f:
            cfg = Config(yaml.safe_load(f))
    """
    mlflow: TrainMLflowConfig
    model: TrainModelConfig
    training: TrainingConfig
    dataset: DatasetConfig

    def __init__(self, yaml_dict: dict):
        """
        Args:
            yaml_dict: dict as returned by yaml.safe_load.
        """
        mlflow_config = TrainMLflowConfig(**yaml_dict['mlflow'])
        model_config = TrainModelConfig(**yaml_dict['model'])
        training_config = TrainingConfig(**yaml_dict['training'])
        dataset_config = DatasetConfig(
            train = TrainDatasetConfig(**yaml_dict['dataset']['train']), 
            test = TestDatasetConfig(**yaml_dict['dataset']['test']))
        
        super().__init__(
            mlflow = mlflow_config,
            model = model_config,
            training = training_config,
            dataset = dataset_config
        )

class EvalConfig(BaseModel):
    """
    Top-level config object for evaluation, populated from eval-config.yaml.

    Expected YAML structure::

        mlflow:
          experiment: congress-bill-classifier
          
        model:
          model_id:         # optional; omit or leave blank to load best model

        dataset:
          test:
            start_date: 2026-01-01
            end_date:         # omit or leave blank to default to datetime.now()

    Usage::

        import yaml
        from config import EvalConfig

        with open("eval-config.yaml") as f:
            cfg = EvalConfig(yaml.safe_load(f))
    """
    mlflow: EvalMLflowConfig
    model: EvalModelConfig
    dataset: DatasetConfig

    def __init__(self, yaml_dict: dict):
        super().__init__(
            mlflow = EvalMLflowConfig(**yaml_dict["mlflow"]),
            model = EvalModelConfig(**yaml_dict["model"]),
            dataset = DatasetConfig(
                test = TestDatasetConfig(**yaml_dict["dataset"]["test"]),
            ),
        )