from airflow.sdk import dag, task, Param
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.providers.standard.sensors.time_delta import TimeDeltaSensor
from airflow.models import DagRun
import pandas as pd

from datetime import datetime, timedelta
import logging
import os
import asyncio

from pipeline import extract as ex, transform as tf, load as ld
from pipeline.tracking import utils

default_args = {
    "owner": "JustinP",
    "retries": 5,
    "retry_delta": timedelta(seconds = 2)
}

@dag(
    dag_id = "pipeline_start",
    description = "Performs initial setup for pipeline by idenitfying which bills to pull. " \
    "Writes bill ids to queue file.",
    tags = ["congress_pipeline"],
    default_args = default_args,
    params={
        "congress_num": Param(default = 118, type = "integer", description = "Congress number to fetch"),
        "batch_size": Param(default = 250, type = "integer", description = "Number of bills to extract per batch"),
        "mode": Param(default = "incremental", enum = ["incremental", "full"],
                      description = "Specifies whether to pull all bills from the specified congress, "
                      "or only the bills updated since the specified number of `weeks_back`."),
        "weeks_back": Param(default = 1, type = "integer", 
                            description = "Only used in incremental mode. Number of weeks back to fetch updated bills.")
       },
    start_date = datetime(2026, 3, 5),
    schedule = "@weekly",
    catchup = False,
    max_active_runs = 1)
def pipeline_start():
    @task()
    def get_start_date(**context) -> datetime | None:
        mode = context["params"]["mode"]
        if mode == "full":
            return None
        elif mode == "incremental":
            weeks_back = context["params"]["weeks_back"]
            return context["logical_date"] - timedelta(days = 7*weeks_back)

    @task()
    def get_bills(last_run_date: datetime | None, **context):
        congress_num = context["params"]["congress_num"]
        async def _get_bills():
            api_key = os.getenv('API_KEY')
            client = ex.CongressAPIClient(api_key)

            # get bills updated since last run
            bill_ids = ex.get_bill_ids(client, congress_num, last_run_date)
            queue_df = utils.generate_queue(congress_num, bill_ids)

            # get bills that pipeline failed on prev run
            if utils.failures_exist(congress_num):
                prev_fails = utils.read_failures(congress_num)
                prev_fails = utils.reset_statuses(prev_fails)
                queue_df = pd.concat([prev_fails, queue_df])
        asyncio.run(_get_bills())


    exit_dag = TriggerDagRunOperator(
        task_id = "exit_dag",
        trigger_dag_id = "etl_bills",
        conf = {"congress_num": "{{ params.congress_num }}",
                "batch_size": "{{ params.batch_size }}"}
    )

    get_date_task = get_start_date()
    write_bills_task = get_bills(get_date_task)
    
    write_bills_task >> exit_dag


@dag(
    dag_id = "pipeline_run",
    description = "Performs ETL tasks.",
    tags = ["congress_pipeline"],
    default_args = default_args,
    params={
       "congress_num": Param(default = 118, type = "integer", description = "Congress number to fetch"),
       "batch_size": Param(default = 250, type = "integer", description = "Number of bills to extract per batch"),
       # last two params are handled by TriggerDagRunOperators
       "rate_limit_retries": Param(default = 0, type = "integer"),
       "pull_members": Param(default = False, type = "boolean", description = "Indicates whether to pull member data")
       },
    start_date = datetime(2026, 3, 5),
    schedule = None,
    catchup = False,
    max_active_runs = 1)
def pipeline_run():

    @task()
    def etl_members(**context):
        pull_members = context["params"]["pull_members"]
        if not pull_members:
            return

        congress_num = context["params"]["congress_num"]
        async def _batch_etl_bills():
            api_key = os.getenv('API_KEY')
            client = ex.CongressAPIClient(api_key)
            raw_members = await ex.extract_members(client, congress_num)
            clean_members = tf.transform_members(congress_num, raw_members)
            ld.upsert_members(clean_members)
        asyncio.run(_batch_etl_bills())

    @task.branch()
    def check_queue_state(**context):
        congress_num = context["params"]["congress_num"]
        queue_df = utils.read_queue(congress_num)
        # only checking extract layer has attempted all bills
        queue_state = utils.get_status_counts(queue_df, layer = "Extract")
        unattempted = queue_state['unattempted']
        if unattempted == 0:
            return "exit_dag"
        else:
            return "etl_bills"

    exit_dag = TriggerDagRunOperator(
        task_id = "exit_dag",
        trigger_dag_id = "clean_up",
        conf = {"congress_num": "{{ params.congress_num }}"}
    )

    @task()
    def batch_etl_bills(**context):
        congress_num = context["params"]["congress_num"]
        batch_size = context["params"]["batch_size"]
        queue_df = utils.read_queue(congress_num)

        async def _batch_etl_bills():
            api_key = os.getenv('API_KEY')
            client = ex.CongressAPIClient(api_key)
            
            # extract
            result = await ex.batch_extract_bill_info(client, queue_df, batch_size)
            raw_bills = result.bills
            
            # transform 
            clean_bills = tf.transform_bills(congress_num, raw_bills)
            clean_sponsorships = tf.transform_bill_sponsorships(raw_bills)

            # load 
            ld.upsert_bills(congress_num, clean_bills)
            ld.upsert_sponsorships(congress_num, clean_sponsorships)

            return result.rate_limited  # indicates whether or not to rerun immediately or wait

        return asyncio.run(_batch_etl_bills())  # push indicator to XComs

    @task.branch()
    def check_rate_limit(rate_limited: bool, **context):
        max_rate_limit_retries = 2
        current_retries = context["params"]["rate_limit_retries"]

        if rate_limited:
            if current_retries >= max_rate_limit_retries:
                return "max_rate_limit_exceeded"
            else:
                return "wait_for_rate_limit"
        else:
            return "retrigger"

    wait_for_rate_limit = TimeDeltaSensor(
        task_id = "wait_for_rate_limit",
        delta = timedelta(minutes = 30),
        mode = "reschedule",
        deferrable = True,
        poke_interval = 60
    )

    retrigger = TriggerDagRunOperator(
        task_id = "retrigger",
        trigger_dag_id = "etl_bills",
        conf = {"congress_num": "{{ params.congress_num }}",
                "batch_size": "{{ params.batch_size }}",
                "rate_limit_retries": 0, # reset num retries if DAG was not slept
                "pull_members": "false"
                }
    )

    retrigger_after_sleep = TriggerDagRunOperator(
        task_id = "retrigger_after_sleep",
        trigger_dag_id = "etl_bills",
        conf = {"congress_num": "{{ params.congress_num }}",
                "batch_size": "{{ params.batch_size }}",
                "rate_limit_retries": "{{ params.rate_limit_retries | int + 1 }}",   # rate limit hit, so increment num retries by one
                "pull_members": "false"
                }
    )

    members_task = etl_members()
    queue_check = check_queue_state()
    bill_etl = batch_etl_bills()
    rate_check = check_rate_limit(bill_etl)

    queue_check >> [exit_dag, bill_etl]
    bill_etl >> rate_check >> [wait_for_rate_limit, retrigger]
    wait_for_rate_limit >> retrigger_after_sleep
    

@dag(
    dag_id = "pipeline_cleanup",
    description = "Records failures from queue file and deletes queue.",
    tags = ["congress_pipeline"],
    default_args = default_args,
    params={
       "congress_num": Param(default = 118, type = "integer", description = "Congress number to fetch"),
       },
    start_date = datetime(2026, 3, 5),
    schedule = None,
    catchup = False,
    max_active_runs = 1)
def pipeline_cleanup():
    pass

pipeline_segment_1 = pipeline_start()
pipeline_segment_2 = pipeline_run()
pipeline_segment_3 = pipeline_cleanup()