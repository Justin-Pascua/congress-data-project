import pandas as pd

import os
from pathlib import Path
import time
from typing import List, Optional, Literal
import logging

from .api_client import CongressAPIClient
from .status import ExtractStatus
from ..exceptions import *
from ..transform.enums import BillType
from ..tracking import utils
from ..tracking.status import ExtractStatus, TransformStatus, LoadStatus

# idea:
# - extract layer calls the api to get the bill types and bill numbers for the bills that the user wants to extract
# - extract layer writes these bill types/numbers into a temp file
# - extract layer extracts bill info until failure (i.e. until rate limit is exceeded) and updates the temp file to indicate which bills have been extracted
# - after an hour, continue running extraction until failure
# - continue until all bills have been extracted

logger = logging.getLogger("pipeline.extract")

VALID_BILL_TYPES = [item.value for item in BillType]

async def extract_members(client: CongressAPIClient, congress_num: int) -> list:
    """
    Returns list of representatives in specified congress
    Args:
        client: a `CongressAPIClient` instance used to make requests to the API
        congress_num: the number of the congress (e.g. 119)
    """
    # get all reps
    members = await client.get_all_members(congress_num)
    logger.info(f"Extracted {len(members)} members in congress {congress_num}")
    return members

async def get_bill_ids(client: CongressAPIClient, congress_num: int = None) -> List[tuple]:
    """
    Returns a list of bill identifiers, whose elements are tuples of the form (`bill_type`, `bill_num`)
    Args:
        client: a `CongressAPIClient` instance used to make requests to the API
        congress_num: the number of the congress (e.g. 119)
    """
    logger.info(f"Fetching bill ids for congress {congress_num}")
    if congress_num is None:
        current_details = await client.get_current_congress()
        congress_num = current_details['congress_num']

    bill_ids = []
    for bill_type in VALID_BILL_TYPES:
        bills_of_type = await client.get_all_bills(congress_num, bill_type)
        current_bill_ids = [(bill_type, bill['number']) for bill in bills_of_type]
        bill_ids.extend(current_bill_ids)
    logger.info(f"Identified {len(bill_ids)} bills in congress {congress_num}")

    return bill_ids

async def single_extract_bill_info(client: CongressAPIClient, 
                                   congress_num: int, bill_type: str, bill_num: int) -> dict:
    """
    Returns the info, summary, and cosponsors of a specified congressional bill.
    Args:
        client: a `CongressAPIClient` instance used to make requests to the API
        congress_num: the number of the congress (e.g. 119)
        bill_type:
        bill_num:
    """
    bill_info = await client.get_bill_info(congress_num, bill_type, bill_num)            
    bill_summary = await client.get_bill_summary(congress_num, bill_type, bill_num)
    cosponsors = await client.get_bill_cosponsors(congress_num, bill_type, bill_num)

    result = {'bill': bill_info,
              'summary': bill_summary,
              'cosponsors': cosponsors}
    
    return result

async def batch_extract_bill_info(client: CongressAPIClient, congress_num: int, ledger_df: pd.DataFrame, 
                                  limit: int = 250, update_ledger: bool = True) -> List[dict]:
    """
    Returns list of bill info in a specified congress using bill identifiers specified by `progress_df`. 
    Note that extraction will stop if the API rate limit is reached.
    Args:
        client: a `CongressAPIClient` instance used to make requests to the API
        congress_num: the number of the congress (e.g. 119)
        ledger_df: a `pd.DataFrame` as outputted by the `read_ledger` function
        limit: the max number of bills to extract
        update_ledger: a bool specifiyng whether or not to call `update_ledger` to update the progress file. 
        If `True`, then `update_ledger` is called. Otherwise, `update_ledger` is not called.
    """

    start_time = time.perf_counter()
    
    # log initial state of ledger
    logger.info(f"Starting batch extraction for congress {congress_num}")
    current_state = utils.get_status_counts(ledger_df = ledger_df, layer = "Extract")
    logger.info(f"Initial state - Total: {current_state['total']} | "
                 f"Unattempted: {current_state['unattempted']} | "
                 f"Successful: {current_state['successful']} | "
                 f"Failed: {current_state['failed']}"
                 )


    result = []
    mask = (ledger_df['Extract Status'] != ExtractStatus.SUCCESSFUL.value)
    bills_to_fetch = ledger_df[mask]

    for i, row_num in enumerate(bills_to_fetch.index):
        if i >= limit:
            logger.info(f"Batch limit of {limit} reached. Stopping.")
            break
        if (i+1) % 25 == 0:
            logger.info(f"Batch extract progress: {i+1} attempted")

        row = bills_to_fetch.iloc[row_num]
        bill_type, bill_num = row[['Bill Type', 'Bill Number']]
        
        try:
            current_item = await single_extract_bill_info(client, congress_num, bill_type, bill_num)
            result.append(current_item)
            bills_to_fetch.at[row_num, "Extract Status"] = ExtractStatus.SUCCESSFUL.value
        except RateLimitError as e:
            logger.warning(f"{str(e)}")
            break
        except Exception as e:
            logger.warning(f"Extract failed for bill {congress_num, bill_type, bill_num} | Error: ({type(e)}) {e}")
            bills_to_fetch.at[row_num, "Extract Status"] = ExtractStatus.FAILED.value
            bills_to_fetch.at[row_num, "Error"] = str(e)

    # update ledger df
    ledger_df[mask] = bills_to_fetch
    if update_ledger:
        utils.update_ledger(congress_num, ledger_df)

    # log final state of ledger
    current_state = utils.get_status_counts(ledger_df = ledger_df, layer = "Extract")
    end_time = time.perf_counter()
    logger.info(f"Final state - Total: {current_state['total']} | "
                 f"Unattempted: {current_state['unattempted']} | "
                 f"Successful: {current_state['successful']} | "
                 f"Failed: {current_state['failed']} "
                 f"({end_time - start_time:.2f}s)"
                 )

    return result
