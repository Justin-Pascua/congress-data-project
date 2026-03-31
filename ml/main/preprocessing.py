from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from datetime import datetime

from database.read import read_bills
from ..utils.data import process_bills, BillDataset, get_dataloader

def training_data_pipeline(tokenizer, 
                           simplify: bool,
                           train_start_date: datetime, train_end_date: datetime, 
                           test_start_date: datetime, test_end_date: datetime = datetime.now(),
                           weighted_sampling: bool = False,
                           val_frac: float = 0.2,
                           max_length: int = None,
                           **kwargs) -> dict:
    """
    Pulls bill records from the database and constructs train, validation, and
    test DataLoaders using date-based splits to prevent data contamination.
    Training and validation bills are drawn from [train_start_date, train_end_date]
    and split by val_frac. Test bills are drawn from a separate, later date range
    [test_start_date, test_end_date] with no overlap with the training window.

    Args:
        tokenizer: HuggingFace tokenizer used to tokenize and pad sequences within each batch.
        simplify: if `True`, then bill policy areas will be binned into 8 possible classes (as opposed to the original 33).
            If `False`, then the policy areas are left as is. 
        train_start_date: start date for querying training bills (inclusive).
        train_end_date: end date for querying training bills (inclusive).
        test_start_date: start date for querying test bills (inclusive).
        test_end_date: end date for querying test bills (inclusive). Defaults to now.
        weighted_sampling: if `True`, then the `DataLoader` for the training set is 
            created with a `WeightedRandomSampler` weighted by reciprocal of class counts
        val_frac: fraction of the training pool to reserve for validation. Defaults to 0.2.
        max_length: maximum token length for truncating sequences. If None, no truncation is applied.
        **kwargs: additional keyword arguments forwarded to get_dataloader (e.g. batch_size, num_workers).

    Returns:
        dict: A dictionary with keys 'train', 'val', and 'test', each containing
            the corresponding DataLoader.
    """
    if train_start_date > train_end_date:
        raise ValueError('train_start_date should be <= train_end_date')
    if test_end_date is not None:
        if test_start_date > test_end_date:
            raise ValueError('test_start_date should be <= test_end_date')
    if test_start_date <= train_end_date:
        raise ValueError('test_end_date should be >= train_end_date to prevent data contamination')
     
    train_val_bills = read_bills(
        start_date = train_start_date,
        end_date = train_end_date)
    train_bills, val_bills = train_test_split(train_val_bills, test_size = val_frac)
    
    test_bills = read_bills(
        start_date = test_start_date,
        end_date = test_end_date)
    
    train_df = process_bills(train_bills, simplify)
    val_df = process_bills(val_bills, simplify)
    test_df = process_bills(test_bills, simplify)

    train_dataset = BillDataset(train_df, 'summary', 'numericalLabel')
    val_dataset = BillDataset(val_df, 'summary', 'numericalLabel')
    test_dataset = BillDataset(test_df, 'summary', 'numericalLabel')

    train_dataloader = get_dataloader(train_dataset, tokenizer, max_length, weighted_sampling, **kwargs)
    val_dataloader = get_dataloader(val_dataset, tokenizer, max_length, **kwargs)
    test_dataloader = get_dataloader(test_dataset, tokenizer, max_length, **kwargs)

    return {'train': train_dataloader,
            'val': val_dataloader,
            'test': test_dataloader}

def eval_data_pipeline(tokenizer, 
                       simplify: bool,
                       test_start_date: datetime, test_end_date: datetime = datetime.now(),
                       **kwargs) -> DataLoader:
    """
    Pulls bill records from the database and constructs a DataLoader to be used for evaluation.

    Args:
        tokenizer: HuggingFace tokenizer used to tokenize and pad sequences within each batch.
        simplify: if `True`, then bill policy areas will be binned into 8 possible classes (as opposed to the original 33).
            If `False`, then the policy areas are left as is. 
        test_start_date: start date for querying test bills (inclusive).
        test_end_date: end date for querying test bills (inclusive). Defaults to now.
        max_length: maximum token length for truncating sequences. If None, no truncation is applied.
        **kwargs: additional keyword arguments forwarded to get_dataloader (e.g. batch_size, num_workers).
    """
    if test_end_date is not None:
        if test_start_date > test_end_date:
            raise ValueError('test_start_date should be <= test_end_date')
     
    test_bills = read_bills(
        start_date = test_start_date,
        end_date = test_end_date)
    test_df = process_bills(test_bills, simplify)

    test_dataset = BillDataset(test_df, 'summary', 'numericalLabel')

    test_dataloader = get_dataloader(test_dataset, tokenizer, **kwargs)

    return test_dataloader
