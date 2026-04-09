from airflow.sdk import dag, task, Param
from airflow.exceptions import AirflowFailException
from airflow.providers.smtp.operators.smtp import EmailOperator

import mlflow
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime, timedelta
import logging
import os

from ml.main.eval import eval_main, EvalConfig

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
        "start_date": Param(default = None, type = ["null", "string"], format = "date", 
                            description = "Start date for querying test bills. " \
                            "Note that newer bills tend to have null summaries " \
                            "and policy areas, and such bills get dropped by the " \
                            "preprocessing pipeline. If none provided, then this "
                            "is set to 4 weeks back from the run date."), 
        "email": Param(default = os.getenv("AIRFLOW_DEFAULT_EMAIL_TO"), type = ["null", "string"],
                       description = "Specifies where to send summary email to.")
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
        
        start_date = context["params"]["start_date"] 
        if start_date is None:
            start_date = context["logical_date"] - timedelta(days = 28)
        
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
                    "start_date": start_date,
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
        run = mlflow.get_run(mlflow_run_id)

        dt = timedelta(milliseconds = run.info.end_time - run.info.start_time)
        dur = str(dt).split('.')[0]
        start_date = run.data.params["test_start_date"]
        end_date = run.data.params["test_end_date"]
        # mlflow records None as string
        if end_date == 'None' or end_date is None:
            end_date = context["logical_date"].date()

        body = (
            f"<h3>Eval Run Summary ({context['logical_date'].date()})</h3>"
            f"<b>Duration (HH:MM:SS): </b><br>"
            f"- {dur}<br>"
            f"<b>Date range evaluated on:</b><br>"
            f"- Start date: {start_date}<br>"
            f"- End date: {end_date}<br>"
            f"<b>Metrics:</b><br>"
            f"- Accuracy: {round(run.data.metrics['final_test_accuracy'], 3)}<br>"
            f"- F1: {round(run.data.metrics['final_test_f1'], 3)}<br>"
        )
        
        EmailOperator(
            task_id = "email_task",
            to = context["params"]["email"],
            from_email = "airflow@example.com",
            subject = f"Congress Data Project - Evaluation Summary ({context["logical_date"].date()})",
            html_content = body
        ).execute(context)

    experiment_task = verify_experiment()
    model_task = verify_model(experiment_task)
    eval_task = evaluate_model()
    email_task = send_email(eval_task)

    experiment_task >> model_task >> eval_task >> email_task


@dag(
    dag_id = "rebuild_api_image",
    description = "Saves best MLflow-logged model into /api directory and rebuilds the model api image",
    tags = ["congress_bill_classifier"],
    default_args = default_args,
    params = {
        "tag": Param(default = None, type = ["null", "string"], 
                     description = "Semantic version given to tag the image with. " \
                     "If none provided, then the most recent version is fetched, " \
                     "and the minor number is incremented by 1")
    }
)
def rebuild_api_image():
    @task
    def get_tag(**context):
        if context["params"]["tag"] is not None:
            return context["params"]["tag"]
        
        # get most recent version (i.e. largest semantic version)
        # increment minor field by one

    @task
    def update_model(**context):
        # load best logged model from MLflow
        # save into api directory
        pass

    @task
    def rebuild(tag, **context):
        # rebuild image
        pass

eval_dag = eval_best()