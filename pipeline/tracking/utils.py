import pandas as pd 

from typing import List, Literal
from pathlib import Path
import os

from .status import ExtractStatus, TransformStatus, LoadStatus

def initialize_ledger(congress_num: int, bill_ids = List[tuple]) -> None:
    """
    Writes a csv file containing the identifiers for congress bills. Column names are Bill Type, Bill Number, and Status.
    The file is written to the directory ./ledger/congress-`congress_num`
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
    df['Error'] = pd.Series(pd.NA, dtype = "string")

    df.set_index(['Congress Number', 'Bill Type', 'Bill Number'], inplace = True)

    root_dir = Path.cwd()
    output_dir = root_dir / "ledger" / f"congress-{congress_num}"
    file_name = 'bill-ids.csv'
    full_file_path = output_dir / file_name
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df.to_csv(full_file_path)

def read_ledger(congress_num: int) -> pd.DataFrame:
    """
    Reads the ledger file containing the bill identifiers, and returns the info in the form of a `pd.DataFrame`. 
    Note that the ledger file must have the path ./ledger/congress-`congress_num`/bill-ids.csv
    Args:
        congress_num: the number of the congress (e.g. 119)
    """
    root_dir = Path.cwd()
    output_dir = root_dir / "ledger" / f"congress-{congress_num}"
    file_name = 'bill-ids.csv'
    full_file_path = output_dir / file_name

    df = pd.read_csv(full_file_path)
    df.set_index(['Congress Number', 'Bill Type', 'Bill Number'], inplace = True)
    df['Error'] = df['Error'].astype(str)
    return df

def update_ledger(congress_num: int, updated_ledger_df: pd.DataFrame) -> None:
    """
    Updates ledger file given a dataframe containing the updated info
    Args:
        congress_num: the number of the congress (e.g. 119) 
        update_ledger_df: a `pd.DataFrame` containing the updated statuses of the bills
    """
    root_dir = Path.cwd()
    output_dir = root_dir / "ledger" / f"congress-{congress_num}"
    file_name = 'bill-ids.csv'
    full_file_path = output_dir / file_name
    
    updated_ledger_df.to_csv(full_file_path)

def get_status_counts(ledger_df: pd.DataFrame, layer: Literal['Extract', 'Transform', 'Load']) -> dict:
    """
    Computes the number of unattempted, successful, and failed items within a specified pipeline layer.
    Returns a dict with keys ['total', 'unattempted', 'successful', 'failed']
    Args:
        ledger_df: a `pd.DataFrame` containing bill identifiers and their status within the pipeline.
        layer: a string, either Extract, Transform, or Load 
    """
    layer_to_type = {'Extract': ExtractStatus,
                     'Transform': TransformStatus,
                     'Load': LoadStatus}
    
    status_enum = layer_to_type[layer]
    status_col_name = f"{layer} Status"
    
    total_bills = len(ledger_df)
    unattempted_count = len(ledger_df[ledger_df[status_col_name] == status_enum.UNATTEMPTED.value])
    successful_count = len(ledger_df[ledger_df[status_col_name] == status_enum.SUCCESSFUL.value])
    failed_count = len(ledger_df[ledger_df[status_col_name] == status_enum.FAILED.value])

    return {'total': total_bills,
            'unattempted': unattempted_count,
            'successful': successful_count,
            'failed': failed_count}