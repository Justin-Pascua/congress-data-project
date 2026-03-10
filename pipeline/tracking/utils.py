import pandas as pd 

from typing import List, Literal
from pathlib import Path
import os

from .status import ExtractStatus, TransformStatus, LoadStatus

QUEUE_DIR = Path.cwd() / "queue" 

def queue_path(congress_num: int) -> Path:
    """
    Returns path to queue file for a given congress
    Args:
        congress_num: the number of the congress (e.g. 119)
    """
    return QUEUE_DIR / f"congress-{congress_num}/bill_queue.csv"

def queue_exists(congress_num: int) -> bool:
    """
    Checks if the queue file for a given congress_num exists in the expected directory. 
    Args:
        congress_num: the number of the congress (e.g. 119)
    """
    return os.path.exists(queue_path(congress_num))

def generate_queue(congress_num: int, bill_ids = List[tuple]) -> pd.DataFrame:
    """
    Returns a dataframe containing the identifiers for congress bills. 
    Args:
        congress_num: the number of the congress (e.g. 119)
        bill_ids: a list of tuples of the form (`bill type`, `bill number`)
    """

    df = pd.DataFrame(bill_ids, columns = ['Bill Type', 'Bill Number'])
    df['Congress Number'] = congress_num
    df['Extract Status'] = ExtractStatus.UNATTEMPTED.value
    df['Transform Status'] = TransformStatus.UNATTEMPTED.value
    df['Load Status'] = LoadStatus.UNATTEMPTED.value

    # column to store potential error messages
    df['Error'] = pd.Series(pd.NA, dtype = str)

    df.set_index(['Congress Number', 'Bill Type', 'Bill Number'], inplace = True)

    return df

def read_queue(congress_num: int) -> pd.DataFrame:
    """
    Reads the queue file containing the bill identifiers, and returns the info in the form of a `pd.DataFrame`. 
    Note that the queue file must have the path ./queue/congress-`congress_num`/bill_queue.csv
    Args:
        congress_num: the number of the congress (e.g. 119)
    """
    df = pd.read_csv(queue_path(congress_num))
    df.set_index(['Congress Number', 'Bill Type', 'Bill Number'], inplace = True)
    df['Error'] = df['Error'].astype(str)
    return df

def commit_queue(congress_num: int, updated_queue_df: pd.DataFrame) -> None:
    """
    Writes queue file given a dataframe containing the updated info
    Args:
        congress_num: the number of the congress (e.g. 119) 
        update_queue_df: a `pd.DataFrame` containing the updated statuses of the bills
    """
    # ensure that directory exists
    os.makedirs(QUEUE_DIR / f"congress-{congress_num}", exist_ok = True)

    updated_queue_df.to_csv(queue_path(congress_num))

def remove_queue_file(congress_num: int) -> None:
    os.remove(queue_path(congress_num))

def reset_statuses(bills_df) -> pd.DataFrame:
    bills_df['Extract Status'] = ExtractStatus.UNATTEMPTED.value
    bills_df['Transform Status'] = TransformStatus.UNATTEMPTED.value
    bills_df['Load Status'] = LoadStatus.UNATTEMPTED.value
    bills_df['Error'] = pd.Series(pd.NA, dtype = str)
    return bills_df

def get_status_counts(queue_df: pd.DataFrame, layer: Literal['Extract', 'Transform', 'Load']) -> dict:
    """
    Computes the number of unattempted, successful, and failed items within a specified pipeline layer.
    Returns a dict with keys ['total', 'unattempted', 'successful', 'failed']
    Args:
        queue_df: a `pd.DataFrame` containing bill identifiers and their status within the pipeline.
        layer: a string, either Extract, Transform, or Load 
    """
    layer_to_type = {'Extract': ExtractStatus,
                     'Transform': TransformStatus,
                     'Load': LoadStatus}
    
    status_enum = layer_to_type[layer]
    status_col_name = f"{layer} Status"
    
    total_bills = len(queue_df)
    unattempted_count = len(queue_df[queue_df[status_col_name] == status_enum.UNATTEMPTED.value])
    successful_count = len(queue_df[queue_df[status_col_name] == status_enum.SUCCESSFUL.value])
    failed_count = len(queue_df[queue_df[status_col_name] == status_enum.FAILED.value])

    return {'total': total_bills,
            'unattempted': unattempted_count,
            'successful': successful_count,
            'failed': failed_count}

def failures_path(congress_num: int) -> Path:
    return QUEUE_DIR / f"congress-{congress_num}/bill_failures.csv"

def failures_exist(congress_num: int) -> bool:
    return os.path.exists(failures_path(congress_num))

def read_failures(congress_num: int) -> pd.DataFrame | None:
    df = pd.read_csv(failures_path(congress_num))
    df.set_index(['Congress Number', 'Bill Type', 'Bill Number'], inplace = True)
    df['Error'] = df['Error'].astype(str)
    return df

def record_failures(queue_df: pd.DataFrame) -> None:
    congress_num = queue_df.index[0][0]
    failures_df = queue_df[(queue_df['Extract Status'] == ExtractStatus.FAILED.value) | 
                            (queue_df['Transform Status'] == TransformStatus.FAILED.value) | 
                            (queue_df['Load Status'] == LoadStatus.FAILED.value)]
    # ensure that directory exists
    os.makedirs(QUEUE_DIR / f"congress-{congress_num}", exist_ok = True)
    failures_df.to_csv(failures_path(congress_num))

def remove_failures_file(congress_num: int) -> None:
    os.remove(failures_path(congress_num))