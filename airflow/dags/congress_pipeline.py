from airflow.sdk import dag, task, Param
from datetime import datetime, timedelta
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

    @task(trigger_rule = "none_failed_min_one_success")
    def batch_etl_bills(**context):
        congress_num = context["params"]["congress_num"]
        batch_size = context["params"]["batch_size"]
        num_batches = context["params"]["num_batches"]
        ledger_df = utils.read_ledger(congress_num)
        async def _batch_etl_bills():
            api_key = os.getenv('API_KEY')
            client = ex.CongressAPIClient(api_key)
            
            for i in range(num_batches):
                print("="*25, f"Batch {i+1}", "="*25)
                # extract
                raw_bills = await ex.batch_extract_bill_info(client, ledger_df, batch_size)

                # transform 
                clean_bills = tf.transform_bills(congress_num, raw_bills)
                clean_sponsorships = tf.transform_bill_sponsorships(raw_bills)

                # load 
                ld.upsert_bills(congress_num, clean_bills)
                ld.upsert_sponsorships(congress_num, clean_sponsorships)

        asyncio.run(_batch_etl_bills())

    etl_members_task = etl_members()
    branch = check_ledger_exists()
    create_ledger_task = create_ledger()
    batch_etl_bills_task = batch_etl_bills()

    [etl_members_task, branch] 
    branch >> [create_ledger_task, batch_etl_bills_task]
    create_ledger_task >> batch_etl_bills_task

pipeline_dag = congress_pipeline()