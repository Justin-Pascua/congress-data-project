from airflow.sdk import dag, task, Param
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
    dag_id = "congress_pipeline",
    default_args = default_args,
    params={
       "congress_num": Param(default = 118, type = "integer", description = "Congress number to fetch"),
       "batch_size": Param(default = 250, type = "integer", description = "Number of bills to extract per batch"),
       "num_batches": Param(default = 1, type = "integer", description = "Number of batches to extract"),
    },
    start_date = datetime(2026, 3, 5),
    schedule = "@daily",
    catchup = False)
def congress_pipeline():

    @task()
    def etl_members(**context):
        congress_num = context["params"]["congress_num"]
        async def _etl_members():
            api_key = os.getenv('API_KEY')
            client = ex.CongressAPIClient(api_key)

            raw_members = await ex.extract_members(client, congress_num)
            clean_members = tf.transform_members(congress_num, raw_members)
            ld.upsert_members(clean_members)

        asyncio.run(_etl_members())

    @task.branch()
    def check_ledger_exists(**context):
        congress_num = context["params"]["congress_num"]
        if utils.ledger_exists(congress_num):
            return "batch_etl_bills"
        else:
            return "create_ledger"

    @task()
    def create_ledger(**context):
        congress_num = context["params"]["congress_num"]
        async def _create_ledger():
            api_key = os.getenv('API_KEY')
            client = ex.CongressAPIClient(api_key)

            bill_ids = await ex.get_bill_ids(client, congress_num)
            utils.initialize_ledger(congress_num, bill_ids)
        asyncio.run(_create_ledger())

    @task()
    def get_batch_indices(**context):
        num_batches = context["params"]["num_batches"]
        return list(range(num_batches))
    
    @task(trigger_rule = "none_failed_min_one_success",
          max_active_tis_per_dag = 1)
    def batch_etl_bills(batch_num, **context):
        congress_num = context["params"]["congress_num"]
        batch_size = context["params"]["batch_size"]
        ledger_df = utils.read_ledger(congress_num)

        async def _batch_etl_bills():
            api_key = os.getenv('API_KEY')
            client = ex.CongressAPIClient(api_key)
            
            print("="*25, f"Batch {batch_num+1}", "="*25)
            # extract
            raw_bills = await ex.batch_extract_bill_info(client, ledger_df, batch_size)

            # transform 
            clean_bills = tf.transform_bills(congress_num, raw_bills)
            clean_sponsorships = tf.transform_bill_sponsorships(raw_bills)

            # load 
            ld.upsert_bills(congress_num, clean_bills)
            ld.upsert_sponsorships(congress_num, clean_sponsorships)

        asyncio.run(_batch_etl_bills())

    @task(trigger_rule = "all_done")
    def summarize_ledger(**context):
        logger = logging.getLogger("airflow.task")
        congress_num = context["params"]["congress_num"]
        ledger_df = utils.read_ledger(118)
        ex_states = utils.get_status_counts(ledger_df, 'Extract')
        tf_states = utils.get_status_counts(ledger_df, 'Transform')
        ld_states = utils.get_status_counts(ledger_df, 'Load')
        ex_states, tf_states, ld_states

        unattempted = ex_states['unattempted']
        ex_failures = ex_states['failed']
        tf_failures = tf_states['failed']
        ld_failures = ld_states['failed']
        successes = ld_states['successful']

        output_str = f"""Final State 
        \t- Unattempted: {unattempted}
        \t- Extract Failures: {ex_failures}
        \t- Transform Failures: {tf_failures}
        \t- Load Failures: {ld_failures}
        \t- Successes: {successes}"""
        logger.info(output_str)

    etl_members_task = etl_members()
    branch = check_ledger_exists()
    create_ledger_task = create_ledger()
    batch_indices = get_batch_indices()
    batch_etl_bills_task = batch_etl_bills.expand(batch_num=batch_indices)
    summary_task = summarize_ledger()

    [etl_members_task, branch] 
    branch >> [create_ledger_task, batch_etl_bills_task]
    create_ledger_task >> batch_etl_bills_task >> summary_task

pipeline_dag = congress_pipeline()