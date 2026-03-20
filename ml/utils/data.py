import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

import re
from typing import List

def strip_html_tags(text: str) -> str:
    """
    Remove HTML tags from a string using regex
    Args:
        text: a string possibly containing html tags
    """
    if not isinstance(text, str):
        return text
    
    # Remove HTML tags
    clean_text = re.sub(r'<[^>]+>', '', text)
    
    # Remove extra whitespace
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    return clean_text

policy_area_mapping = {
    # Security & Defense
    "Armed Forces and National Security": "Security & Defense",
    "International Affairs": "Security & Defense",
    "Emergency Management": "Security & Defense",
    "Crime and Law Enforcement": "Security & Defense",

    # Economy & Finance
    "Taxation": "Economy & Finance",
    "Finance and Financial Sector": "Economy & Finance",
    "Economics and Public Finance": "Economy & Finance",
    "Commerce": "Economy & Finance",
    "Foreign Trade and International Finance": "Economy & Finance",

    # Health & Welfare
    "Health": "Health & Welfare",
    "Social Welfare": "Health & Welfare",
    "Families": "Health & Welfare",
    "Animals": "Health & Welfare",

    # Environment & Energy
    "Environmental Protection": "Environment & Energy",
    "Energy": "Environment & Energy",
    "Water Resources Development": "Environment & Energy",
    "Public Lands and Natural Resources": "Environment & Energy",

    # Society & Civil Rights
    "Civil Rights and Liberties, Minority Issues": "Society & Civil Rights",
    "Immigration": "Society & Civil Rights",
    "Native Americans": "Society & Civil Rights",
    "Social Sciences and History": "Society & Civil Rights",

    # Infrastructure & Industry
    "Transportation and Public Works": "Infrastructure & Industry",
    "Housing and Community Development": "Infrastructure & Industry",
    "Science, Technology, Communications": "Infrastructure & Industry",
    "Agriculture and Food": "Infrastructure & Industry",

    # Education & Culture
    "Education": "Education & Culture",
    "Arts, Culture, Religion": "Education & Culture",
    "Sports and Recreation": "Education & Culture",

    # Government & Law
    "Government Operations and Politics": "Government & Law",
    "Congress": "Government & Law",
    "Law": "Government & Law",
    "Labor and Employment": "Government & Law",
    "Private Legislation": "Government & Law",
}

def policy_area_simplifier(raw_policy_area: str) -> str:
    """
    Maps raw policy areas to simplified classes
    Args:
        raw_polcy_area: a raw policy area string
    """
    try:
        return policy_area_mapping[raw_policy_area]
    except:
        return raw_policy_area
    
class BillDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_col: str | List[str], target_col: str | List[str]):
        self.X = df[feature_col].reset_index(drop = True)
        self.y = df[target_col].reset_index(drop = True)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        # not applying tokenizer here. 
        # delegate tokenization to DataLoader so that tokenizer can handle padding
        x, y = self.X.iloc[index], self.y.iloc[index]
        return x, y
    
def make_collate_fn(tokenizer, max_length: int = None):
    """
    Factory function that creates collate_fn to be passed to a torch `DataLoader` object. 
    Args:
        tokenizer: a HuggingFace tokenizer. This handles the work of padding sequences with zeroes
        max_length: int passed to tokenizer to determine max token length of sequences. 
    """
    def collate_fn(batch):
        features, labels = zip(*batch)
        x = tokenizer(features, 
                      padding = "longest", 
                      truncation = True,
                      max_length = max_length, 
                      return_tensors = "pt")
        y = torch.tensor(labels, dtype = torch.long)
        # HF model expects labels in 'labels' parameter
        return x | {'labels': y}

    return collate_fn

def get_dataloader(dataset: BillDataset, tokenizer, max_length: int = None, **kwargs) -> DataLoader:
    """
    Returns a torch `DataLoader` equipped with a collate_fn as returned by `make_collate_fn`.
    Args:
        dataset: a `BillDataset` object.
        tokenizer: a HuggingFace tokenizer passed to `make_collate_fn` to construct the collate_fn passed to the `DataLoader` constructor.
        max_length: int passed to tokenizer to determine max token length of sequences. 
    """
    
    dataloader = DataLoader(
        dataset, 
        collate_fn = make_collate_fn(tokenizer, max_length),
        **kwargs
    )
    return dataloader

