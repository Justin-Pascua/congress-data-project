from .api_client import CongressAPIClient
from .status import ExtractStatus
from ..exceptions import *
from ..transform.enums import BillType

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

VALID_BILL_TYPES = [item.value for item in BillType]

async def extract_members(client: CongressAPIClient, congress_num: int) -> list:
    """
    Returns list of representatives in specified congress
    Args:
        client: a `CongressAPIClient` instance used to make requests to the API
        congress_num: the number of the congress (e.g. 119)
    """
    # get all reps
    representatives = await client.get_all_members(congress_num)
    return representatives

async def get_bill_ids(client: CongressAPIClient, congress_num: int = None) -> List[tuple]:
    """
    Returns a list of bill identifiers, whose elements are tuples of the form (`bill_type`, `bill_num`)
    Args:
        client: a `CongressAPIClient` instance used to make requests to the API
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
    """
    Updates progress file given a dataframe containing the updated info
    Args:
        congress_num: the number of the congress (e.g. 119) 
        update_progress_df: a `pd.DataFrame` containing the updated extraction statuses of the bills
    """
    output_dir = f'./progress/congress-{congress_num}'
    output_filename = 'bill-ids.csv'
    path = os.path.join(output_dir, output_filename)
    
    updated_progress_df.to_csv(path, index = False)

async def batch_extract_bill_info(client: CongressAPIClient, congress_num: int, progress_df: pd.DataFrame, 
                                  limit: int = 250, log_progress: bool = True) -> list:
    """
    Returns list of bill info in a specified congress using bill identifiers specified by `progress_df`. 
    Note that extraction will stop if the API rate limit is reached.
    Args:
        client: a `CongressAPIClient` instance used to make requests to the API
        congress_num: the number of the congress (e.g. 119)
        progress_df: a `pd.DataFrame` as outputted by the `read_progress` function
        limit: the max number of bills to extract
        log_progress: a bool specifiyng whether or not to call `update_progress` to update the progress file. 
        If `True`, then `update_progress` is called. Otherwise, `update_progress` is not called.
    """
    
    result = []
    
    unattempted_bills = progress_df[progress_df['Status'] == ExtractStatus.UNATTEMPTED.value]

    for i, row_num in enumerate(unattempted_bills.index):
        if i >= limit:
            break

        row = unattempted_bills.iloc[row_num]
        bill_type, bill_num = row[['Bill Type', 'Bill Number']]
        
        try:
            bill_info = await client.get_bill_info(congress_num, bill_type, bill_num)            
            bill_summary = await client.get_bill_summary(congress_num, bill_type, bill_num)
            cosponsors = await client.get_bill_cosponsors(congress_num, bill_type, bill_num)

            unattempted_bills.at[row_num, "Status"] = ExtractStatus.EXTRACTED.value
            current_item = {'bill': bill_info,
                            'summary': bill_summary,
                            'cosponsors': cosponsors}
            result.append(current_item)
        except RateLimitError as e:
            print(e)
            break
        except:
            unattempted_bills.at[row_num, "Status"] = ExtractStatus.FAILED.value

    
    if log_progress:
        progress_df[progress_df['Status'] == ExtractStatus.UNATTEMPTED.value] = unattempted_bills
        await update_progress(congress_num, progress_df)

    return result
