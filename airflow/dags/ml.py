from airflow.sdk import dag, task, Param
from airflow.exceptions import AirflowFailException

import mlflow
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime, timedelta
import logging
import os

from ml.main.eval import eval_main, EvalConfig
from ml.main.train import train_main, TrainConfig

logger = logging.getLogger(__name__)

default_args = {
    "owner": "JustinP",
    "retries": 5,
    "retry_delta": timedelta(seconds = 2)
}

@dag(
    dag_id = "eval_best",
    description = "Evaluates best MLflow-logged model on recent data from pipeline, sends results via email.",
    tags = ["congress_bill_classifier"],
    default_args = default_args,
    params={
        "experiment": Param(default = "congress-bill-classifier", type = "string",
                            description = "Name of MLflow experiment"),
        "description": Param(default = None, type = ["null", "string"],
                             description = "Description given to the MLflow run"),
        "model_id": Param(default = None, type = ["null", "string"], 
                          description = "MLflow-prescribed id of model to be evaluated. " \
                          "If None, then defaults to best model logged in MLflow"),
        "weeks_back": Param(default = 4, type = ["null", "integer"], 
                            description = "Only used in incremental mode. Number of weeks back to " \
                            "fetch updated bills. Note that newer bills tend to have null summaries "
                            "and policy areas, and such bills get dropped by the preprocessing pipeline."), 
       },
    start_date = datetime(2026, 3, 30),
    schedule = "0 0 * * 1",
    catchup = False,
    max_active_runs = 1)
def eval_best():
    @task
    def verify_experiment(**context):
        # verify that the user-provided experiment name actually corresponds to an experiment logged in MLflow
        experiment_name = context["params"]["experiment"]
        experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            raise AirflowFailException(f"Provided experiment name ({experiment_name}) does not correspond to an experiment found in the MLflow logs.")

        return experiment.experiment_id

    @task()
    def verify_model(experiment_id: str, **context):
        model_id = context["params"]["model_id"]
        if model_id is not None:
            logger.info("Verifying that provided model id corresponds to a model logged in MLflow")
            try:
                # raises MLflow exception if model not found
                logged_model = mlflow.get_logged_model(model_id)
            except Exception as e:
                raise AirflowFailException(f"{e}")
        else:
            logger.info("No model id provided. Verifying that any model exists in MLflow logs.")
            model_results = mlflow.search_logged_models(
                experiment_ids = [experiment_id]
            )
            if len(model_results) == 0:
                raise AirflowFailException("No model found in MLflow logs.")

    @task()
    def evaluate_model(**context):
        logger.info("Initializing config")
        config = EvalConfig({
            "mlflow": {
                "experiment": context["params"]["experiment"],
                "description": context["params"]["description"],
                "log_figs": False
            },
            "model": {
                "model_id": context["params"]["model_id"]
            },
            "dataset":{
                "test": {
                    "start_date": context["logical_date"] - timedelta(days = 7*context["params"]["weeks_back"]),
                    "end_date": None
                }
            }
        })

        logger.info("Evaluating model")
        mlflow_run_id = eval_main(config)
        logger.info("Evaluation complete")

        return mlflow_run_id

    @task()
    def send_email(mlflow_run_id, **context):
        logger.info(f"Received run id: {mlflow_run_id}")
        pass

    experiment_task = verify_experiment()
    model_task = verify_model(experiment_task)
    eval_task = evaluate_model()
    email_task = send_email(eval_task)

    experiment_task >> model_task >> eval_task >> email_task

eval_dag = eval_best()

# idea:
# alert DAG:
# - evaluates best model on most recent data
# - sends email if metrics fall below threshold

# retrain DAG:
# - trains base model or resumes model training on more recent data
# - evaluates
# - if threshold not met, send email
# - if threshold met, update Docker image for API and push changes to cloud