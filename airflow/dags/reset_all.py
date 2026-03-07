from airflow.sdk import dag, task, Param
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator

from datetime import datetime, timedelta
import logging

from pipeline.tracking import utils

default_args = {
    "owner": "JustinP",
    "retries": 5,
    "retry_delta": timedelta(seconds = 2)
}

logger = logging.getLogger(__name__)

@dag(
    dag_id = "reset_all",
    description = "USED INTERNALLY FOR DEV TESTING. Wipes all data from database and resets queue file",
    default_args = default_args,
    params={
       "congress_num": Param(default = 118, type = "integer", description = "Congress number to fetch"),
    },
    start_date = datetime(2026, 3, 5),
    schedule = None,
    catchup = False
)
def reset_all():
    @task
    def reset_queue(**context):
        congress_num = context["params"]["congress_num"]
        try:
            queue_df = utils.read_queue(congress_num)
            queue_df = utils.reset_statuses(queue_df)
            utils.commit_queue(congress_num, queue_df)
        except:
            logger.info("Queue file did not exist. No changes made")

    reset_db = SQLExecuteQueryOperator(
        task_id = "reset_db",
        conn_id = "congress-db",
        sql = """
            DELETE FROM bills WHERE congress_num = %(congress_num)s;
            DELETE FROM members WHERE congress_num = %(congress_num)s;
            DELETE FROM bill_sponsorship WHERE congress_num = %(congress_num)s;
        """,
        parameters={"congress_num": "{{ params.congress_num }}"}
    )

    [reset_queue(), reset_db]

resetter_dag = reset_all()