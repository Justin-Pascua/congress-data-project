import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from database.read import read_bills
from ..utils.data import strip_html_tags, policy_area_simplifier, BillDataset, get_dataloader

def training_data_pipeline(tokenizer, bills: list, max_length: int, 
                           test_frac: float = 0.2, val_frac: float = 0.2, **kwargs) -> dict:
    """
    Transforms raw bill records into train, validation, and test `DataLoader` objects ready for model training.
    The pipeline strips HTML tags from bill sumamries, simplifies policy area labels into broader classes, 
    encodes these classes numerically, then splits the data into train/val/test sets. These are passed to DataLoaders which
    are equipped with the provided tokenizer which handles sequence padding.
    Args:
        tokenizer: a HuggingFace tokenizer used to tokenize and pad sequences within each batch.
        bills: a list of bill records pulled from the database, each expected to have a summary and policy_area attribute.
        max_length: max token length for truncating sequences during tokenization.
        test_frac: fraction of the full dataset to reserve for the test set.
        val_frac: fraction of the remaining training data to reserve for validation.
        **kwargs: additional keyword arguments forwarded to get_dataloader (e.g. batch_size, num_workers).

    Returns:
        dict: A dictionary with the following keys:
            - 'dataloaders': dict with 'train', 'val', and 'test' DataLoaders.
            - 'label_encoder': fitted LabelEncoder instance mapping numerical labels back to string labels.
    """
    summaries = [strip_html_tags(bill.summary) for bill in bills]
    labels = [policy_area_simplifier(bill.policy_area) for bill in bills]

    df = pd.DataFrame(list(zip(summaries, labels)), columns = ['summary', 'label'])
    df = df.dropna().reset_index(drop = True)

    encoder = LabelEncoder()
    df['numericalLabel'] = encoder.fit_transform(df['label'])

    train_df, test_df = train_test_split(df, test_size = test_frac, shuffle = True)
    train_df, val_df = train_test_split(train_df, test_size = val_frac, shuffle = True)
    
    train_dataset = BillDataset(train_df, 'summary', 'numericalLabel')
    val_dataset = BillDataset(val_df, 'summary', 'numericalLabel')
    test_dataset = BillDataset(test_df, 'summary', 'numericalLabel')

    train_dataloader = get_dataloader(train_dataset, tokenizer, max_length, **kwargs)
    val_dataloader = get_dataloader(val_dataset, tokenizer, max_length, **kwargs)
    test_dataloader = get_dataloader(test_dataset, tokenizer, max_length, **kwargs)

    return {'dataloaders': 
                {'train': train_dataloader,
                'val': val_dataloader,
                'test': test_dataloader
                },
            'label_encoder': encoder
            }

