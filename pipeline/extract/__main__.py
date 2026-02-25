from .api_client import CongressAPIClient
from .status import ExtractStatus
from ..config import settings
from ..exceptions import *

import pandas as pd

import os
import time
from typing import List, Optional, Literal

# idea:
# - extract layer calls the api to get the bill types and bill numbers for the bills that the user wants to extract
# - extract layer writes these bill types/numbers into a temp file
# - extract layer extracts bill info until failure (i.e. until rate limit is exceeded) and updates the temp file to indicate which bills have been extracted
# - after an hour, continue running extraction until failure
# - continue until all bills have been extracted

# omitted "hjres", "sjres", "hconres", "sconres", "hres", "sres" from valid types because only hr and s can become laws
VALID_BILL_TYPES = ["hr", "s"]

client = CongressAPIClient(settings.API_KEY.get_secret_value())

async def extract_reps(congress_num: int) -> list:
    """
    Returns list of representatives in specified congress
    Args:
        congress_num: the number of the congress (e.g. 119)
    """
    # get all reps
    representatives = await client.get_all_members(congress_num)
    return representatives

async def get_bill_ids(congress_num: int = None) -> List[tuple]:
    """
    Returns a list of bill identifiers, whose elements are tuples of the form (`bill_type`, `bill_num`)
    Args:
        congress_num: the number of the congress (e.g. 119)
    """
    if congress_num is None:
        current_details = await client.get_current_congress()
        congress_num = current_details['congress_num']

    bill_ids = []
    for bill_type in VALID_BILL_TYPES:
        bills_of_type = await client.get_all_bills(congress_num, bill_type)
        current_bill_ids = [(bill_type, bill['number']) for bill in bills_of_type]
        bill_ids.extend(current_bill_ids)

    return bill_ids

async def initialize_progress(congress_num: int, bill_ids = List[tuple]) -> None:
    """
    Writes a csv file containing the identifiers for congress bills. Column names are Bill Type, Bill Number, and Status.
    The file is written to the directory ./progress/congress-`congress_num`
    Args:
        congress_num: the number of the congress (e.g. 119)
        bill_ids: a list of tuples of the form (`bill type`, `bill number`)
    """

    df = pd.DataFrame(bill_ids, columns = ['Bill Type', 'Bill Number'])
    df['Status'] = ExtractStatus.UNATTEMPTED.value

    output_dir = f'./progress/congress-{congress_num}'
    output_filename = 'bill-ids.csv'
    full_path = os.path.join(output_dir, output_filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df.to_csv(full_path, index = False)

async def read_progress(congress_num: int) -> pd.DataFrame:
    """
    Reads the progress file containing the bill identifiers, and returns the info in the form of a `pd.DataFrame`. 
    Note that the progress file must have the path ./progress/congress-`congress_num`/bill-ids.csv
    Args:
        congress_num: the number of the congress (e.g. 119)
    """
    target_dir = f'./progress/congress-{congress_num}'
    file_name = 'bill-ids.csv'
    path = os.path.join(target_dir, file_name)

    df = pd.read_csv(path)
    return df

async def update_progress(congress_num: int, updated_progress_df: pd.DataFrame) -> None:
    
    full_df = await read_progress(congress_num)

    output_dir = f'./progress/congress-{congress_num}'
    output_filename = 'bill-ids.csv'
    path = os.path.join(output_dir, output_filename)
    
    full_df[full_df['Status'] == ExtractStatus.UNATTEMPTED.value] = updated_progress_df

    full_df.to_csv(path, index = False)

async def batch_extract_bill_info(congress_num: int, unattempted_bills: pd.DataFrame, limit: int = 250) -> dict:

    bills_detailed = {type: dict() for type in VALID_BILL_TYPES}
    bill_cosponsors = {type: dict() for type in VALID_BILL_TYPES}

    for i, row_num in enumerate(unattempted_bills.index):
        if i >= limit:
            break

        row = unattempted_bills.iloc[row_num]
        bill_type, bill_num = row[['Bill Type', 'Bill Number']]
        
        try:
            bill_info = await client.get_bill_info(congress_num, bill_type, bill_num)
            bills_detailed[bill_type][bill_num] = bill_info
            
            cosponsors = await client.get_bill_cosponsors(congress_num, bill_type, bill_num)
            bill_cosponsors[bill_type][bill_num] = cosponsors

            unattempted_bills.at[row_num, "Status"] = ExtractStatus.EXTRACTED.value
        except RateLimitError as e:
            print(e)
            break
    
    await update_progress(congress_num, unattempted_bills)

    return {'bills': bills_detailed,
            'cosponsors': bill_cosponsors}
