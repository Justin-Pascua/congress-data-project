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

logger = logging.getLogger(__name__)

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
        "congress_num": Param(default = None, type = ["null", "integer"], description = "Congress number to fetch. If not provided, then the API is called to fetch the current congress."),
        "batch_size": Param(default = 250, type = "integer", description = "Number of bills to extract per batch"),
        "mode": Param(default = "incremental", enum = ["incremental", "full"],
                      description = "Specifies whether to pull all bills from the specified congress, "
                      "or only the bills updated since the specified number of `weeks_back`."),
        "weeks_back": Param(default = 1, type = ["null", "integer"], 
                            description = "Only used in incremental mode. Number of weeks back to fetch updated bills.")
       },
    start_date = datetime(2026, 3, 5),
    schedule = "@weekly",
    catchup = False,
    max_active_runs = 1)
def pipeline_start():
    @task()
    def get_congress(**context) -> int:
        congress_num_param = context["params"]["congress_num"]
        if congress_num_param is not None:
            return congress_num_param
        else:
            async def _get_current_congress():
                api_key = os.getenv('API_KEY')
                client = ex.CongressAPIClient(api_key)
                current_details = await client.get_current_congress()
                await client.close()
                return current_details["congress_num"]
            return asyncio.run(_get_current_congress())

    @task()
    def get_start_date(**context) -> datetime | None:
        mode = context["params"]["mode"]
        if mode == "full":
            return None
        elif mode == "incremental":
            weeks_back = context["params"]["weeks_back"]
            return context["logical_date"] - timedelta(days = 7*weeks_back)

    @task()
    def get_bill_ids(congress_num: int, start_date: datetime | None):
        async def _get_bill_ids():
            api_key = os.getenv('API_KEY')
            client = ex.CongressAPIClient(api_key)

            # get bills updated since last run
            bill_ids = await ex.get_bill_ids(client, congress_num, start_date)
            queue_df = utils.generate_queue(congress_num, bill_ids)

            # get bills that pipeline failed on prev run
            if utils.failures_exist(congress_num):
                prev_fails = utils.read_failures(congress_num)
                prev_fails = utils.reset_statuses(prev_fails)
                queue_df = pd.concat([prev_fails, queue_df])
                utils.remove_failures_file(congress_num)
            
            # save queue to file
            utils.commit_queue(congress_num, queue_df)

            await client.close()

        asyncio.run(_get_bill_ids())

    @task()
    def exit_dag(congress_num: int, **context):
        batch_size = context["params"]["batch_size"]
        
        TriggerDagRunOperator(
            task_id = "exit_dag",
            trigger_dag_id = "pipeline_run",
            conf = {
                "congress_num": congress_num,
                "batch_size": batch_size,
                "rate_limit_retries": 0,
                "pull_members": True
            }
        ).execute(context)

    get_congress_task = get_congress()
    get_date_task = get_start_date()
    write_bills_task = get_bill_ids(get_congress_task, get_date_task)
    exit_task = exit_dag(get_congress_task)
    
    write_bills_task >> exit_task


@dag(
    dag_id = "pipeline_run",
    description = "Performs ETL tasks.",
    tags = ["congress_pipeline"],
    default_args = default_args,
    params={
       "congress_num": Param(default = 119, type = "integer", description = "Congress number to fetch"),
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
        async def _batch_etl_members():
            api_key = os.getenv('API_KEY')
            client = ex.CongressAPIClient(api_key)
            raw_members = await ex.extract_members(client, congress_num)
            clean_members = tf.transform_members(congress_num, raw_members)
            ld.upsert_members(clean_members)
            await client.close()
        asyncio.run(_batch_etl_members())

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
            return "batch_etl_bills"

    @task()
    def exit_dag(**context):
        congress_num = context["params"]["congress_num"]
        TriggerDagRunOperator(
            task_id = "exit_dag",
            trigger_dag_id = "pipeline_cleanup",
            conf = {"congress_num": congress_num}
        ).execute(context)

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

            await client.close()

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

    @task()
    def retrigger(**context):
        congress_num = context["params"]["congress_num"]
        batch_size = context["params"]["batch_size"]
        TriggerDagRunOperator(
            task_id = "retrigger",
            trigger_dag_id = "pipeline_run",
            conf = {"congress_num": congress_num,
                    "batch_size": batch_size,
                    "rate_limit_retries": 0, # reset num retries if DAG was not slept
                    "pull_members": False
                    }
        ).execute(context)

    @task()
    def retrigger_after_sleep(**context):
        congress_num = context["params"]["congress_num"]
        batch_size = context["params"]["batch_size"]
        rate_limit_retries = context["params"]["rate_limit_retries"]
        TriggerDagRunOperator(
        task_id = "retrigger_after_sleep",
        trigger_dag_id = "pipeline_run",
        conf = {"congress_num": congress_num,
                "batch_size": batch_size,
                "rate_limit_retries": rate_limit_retries + 1,   # rate limit hit, so increment num retries by one
                "pull_members": False
                }
        ).execute(context) 

    members_task = etl_members()
    exit_task = exit_dag()
    queue_check = check_queue_state()
    bill_etl = batch_etl_bills()
    rate_check = check_rate_limit(bill_etl)
    retrigger_task = retrigger()
    sleep_retrigger_task = retrigger_after_sleep()

    queue_check >> [exit_task, bill_etl]
    bill_etl >> rate_check >> [wait_for_rate_limit, retrigger_task]
    wait_for_rate_limit >> sleep_retrigger_task
    
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
    @task()
    def summarize(**context):
        congress_num = context["params"]["congress_num"]
        queue_df = utils.read_queue(congress_num)
        e_statuses = utils.get_status_counts(queue_df, 'Extract')
        t_statuses = utils.get_status_counts(queue_df, 'Transform')
        l_statuses = utils.get_status_counts(queue_df, 'Load')
        
        summary_str = f"Total items attempted: {len(queue_df)}\n" \
        f"Extract - Successes: {e_statuses['successful']} | Failures: {e_statuses['failed']}\n" \
        f"Transform - Successes: {t_statuses['successful']} | Failures: {t_statuses['failed']}\n" \
        f"Load - Successes: {l_statuses['successful']} | Failures: {l_statuses['failed']}\n" 
        
        logger.info(summary_str)

    @task()
    def record_errors(**context):
        congress_num = context["params"]["congress_num"]
        queue_df = utils.read_queue(congress_num)
        utils.record_failures(queue_df)

    @task()
    def clear_queue(**context):
        congress_num = context["params"]["congress_num"]
        utils.remove_queue_file(congress_num)

    summary_task = summarize()
    record_task = record_errors()
    clear_task = clear_queue()

    summary_task >> record_task >> clear_task

pipeline_segment_1 = pipeline_start()
pipeline_segment_2 = pipeline_run()
pipeline_segment_3 = pipeline_cleanup()