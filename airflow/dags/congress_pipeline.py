from airflow.sdk import dag, task
from datetime import datetime, timedelta

from pipeline import extract as ex, transform as tf, load as ld
from pipeline.tracking import utils

default_args = {
    "owner": "JustinP",
    "retries": 5,
    "retry_delta": timedelta(seconds = 2)
}

@dag(dag_id = "congress_pipeline",
     default_args = default_args,
     start_date = datetime(2026, 3, 5),
     schedule = "@daily",
     catchup = False)
def congress_pipeline():
    @task()
    def ping():
        pass