from pydantic import BaseModel
from typing import Optional
from datetime import datetime, date

class MLflowConfig(BaseModel):
    """
    MLflow tracking configuration. 
    """
    experiment: str

class ModelConfig(BaseModel):
    """
    Model selection and architecture configuration.
    """
    force_base: bool    # If True, then loads base model. If False, then loads best model logged in MLflow (or base model if no models logged in MLflow) 
    checkpoint: str     # HuggingFace checkpoint of base model
    num_labels: int     # number of classes in the classification task


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
    train: TrainDatasetConfig
    test: TestDatasetConfig

class Config(BaseModel):
    """
    Top-level configuration object, populated from a YAML config file via yaml.safe_load.

    Expected YAML structure::

        mlflow:
          experiment: my-experiment-name

        model:
          num_labels: 2
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
    mlflow: MLflowConfig
    model: ModelConfig
    training: TrainingConfig
    dataset: DatasetConfig

    def __init__(self, yaml_dict: dict):
        """
        Args:
            yaml_dict: dict as returned by yaml.safe_load.
        """
        mlflow_config = MLflowConfig(**yaml_dict['mlflow'])
        model_config = ModelConfig(**yaml_dict['model'])
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