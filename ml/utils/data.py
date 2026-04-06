import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import re
from typing import List
from dataclasses import dataclass

from database.models import Bill

@dataclass 
class BillSample:
    """
    Dataclass representing a single sample, where x is the text summary 
    of a bill and y is the corresponding label. Used when samples are 
    not indexed by parent bill (i.e. when we don't need to keep track of 
    which chunks belong to which parent bill).
    """
    x: str  # text
    y: int  # label

@dataclass
class IndexedBillSample:
    """
    Dataclass representing a single indexed sample, where parent_idx indicates 
    which parent bill the chunk belongs to, x is the text summary, and y is the
    corresponding label. Used when evaluating in inference mode, where we chunk 
    each bill into multiple pieces and want to aggregate predictions across 
    chunks before computing metrics.
    """
    parent_idx: int  # index indicating which parent bill the chunk belongs to
    x: str  # text
    y: int  # label

class BillDataset(Dataset):
    """
    Dataset returning samples as `BillSample` dataclass instances
    """
    def __init__(self, df: pd.DataFrame):
        self.X = df['summary'].reset_index(drop = True)
        self.y = df['numericalLabel'].reset_index(drop = True)
        self.num_chunks = len(df)
        self.num_bills = len(df['parentIndex'].unique())

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        # not applying tokenizer here. 
        # delegate tokenization to DataLoader so that tokenizer can handle padding
        x, y = self.X.iloc[index], self.y.iloc[index]
        return BillSample(x, y)

class IndexedBillDataset(Dataset):
    """
    Dataset returning samples as `IndexedBillSample` dataclass instances
    """
    def __init__(self, df: pd.DataFrame):
        self.parent_idx = df['parentIndex'].reset_index(drop = True)   # column indicating which parent sample each chunk belongs to
        self.X = df['summary'].reset_index(drop = True)
        self.y = df['numericalLabel'].reset_index(drop = True)
        self.num_chunks = len(df)
        self.num_bills = len(df['parentIndex'].unique())

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        # not applying tokenizer here. 
        # delegate tokenization to DataLoader so that tokenizer can handle padding
        parent_idx, x, y = self.parent_idx.iloc[index], self.X.iloc[index], self.y.iloc[index]
        return IndexedBillSample(parent_idx, x, y)

# mapping from raw policy areas to simplified ones for the 8-label version of the task
raw2simplified = {
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

# encoders used to pass labels to other scripts and to decode
possible_raw_labels = list(set(raw2simplified.keys()))
raw_encoder = LabelEncoder()
raw_encoder.fit(possible_raw_labels)

possible_simplified_labels = list(set(raw2simplified.values()))
simplified_encoder = LabelEncoder()
simplified_encoder.fit(possible_simplified_labels)

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

def chunk_text(text: str, tokenizer, overlap: int = 50) -> list[str]:
    """
    Split text into chunks that fit within the tokenizer's max sequence length.
    Args:
        text: the input text to chunk.
        tokenizer: a HuggingFace tokenizer instance.
        overlap: number of tokens to overlap between chunks.
    """
    max_length = tokenizer.model_max_length
    special_tokens_count = tokenizer.num_special_tokens_to_add(pair=False)
    effective_max = max_length - special_tokens_count

    if overlap >= effective_max:
        raise ValueError(
            f"overlap ({overlap}) must be less than effective max length ({effective_max})"
        )

    token_ids = tokenizer.encode(text, add_special_tokens=False)

    if len(token_ids) <= effective_max:
        return [text]

    chunks = []
    step = effective_max - overlap
    start = 0

    while start < len(token_ids):
        end = start + effective_max
        chunk_ids = token_ids[start:end]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens = True)
        chunks.append(chunk_text)
        start += step

    return chunks

def chunk_dataframe(df: pd.DataFrame, tokenizer, overlap: int = 50,
                    batch_size: int = 64) -> pd.DataFrame:
    """
    Chunk a dataframe of text samples so each chunk fits within the tokenizer's
    max sequence length, using batch tokenization for efficiency.

    Args:
        df: Input dataframe with text and label columns.
        tokenizer: A HuggingFace tokenizer instance.
        overlap: Number of tokens to overlap between consecutive chunks.
        text_col: Column name for the input text.
        label_col: Column name for the string label.
        numerical_label_col: Column name for the integer label.
        batch_size: Number of texts to tokenize per batch.
    """
    max_length = tokenizer.model_max_length
    special_tokens_count = tokenizer.num_special_tokens_to_add(pair = False)
    effective_max = max_length - special_tokens_count

    if overlap >= effective_max:
        raise ValueError(
            f"Overlap ({overlap}) must be less than effective max length ({effective_max})"
        )

    texts = df['summary'].tolist()
    labels = df['label'].tolist()
    numerical_labels = df['numericalLabel'].tolist()

    # tokenize text column in batches
    all_token_ids: list[list[int]] = []
    for batch_start in range(0, len(texts), batch_size):
        batch = texts[batch_start : batch_start + batch_size]
        encoded = tokenizer(
            batch,
            add_special_tokens = False,
            truncation = False,
            padding = False,
            return_attention_mask = False,
            return_token_type_ids = False,
        )
        all_token_ids.extend(encoded["input_ids"])

    step = effective_max - overlap

    # slice tokenized samples into chunks and track which label each chunk belongs to
    chunk_token_ids: list[list[int]] = []
    group_idx: list[int] = []
    chunk_labels: list[str] = []
    chunk_numerical_labels: list[int] = []
    for i, (token_ids, label, numerical_label) in enumerate(zip(all_token_ids, labels, numerical_labels)):
        # if sample fits in context window, keep as is
        if len(token_ids) <= effective_max:
            chunk_token_ids.append(token_ids)
            chunk_labels.append(label)
            chunk_numerical_labels.append(numerical_label)
            group_idx.append(i)
        # if sample too long, slice into chunks with specified overlap
        else:
            for start in range(0, len(token_ids), step):
                chunk = token_ids[start : start + effective_max]
                chunk_token_ids.append(chunk)
                chunk_labels.append(label)
                chunk_numerical_labels.append(numerical_label)
                group_idx.append(i)

    # decode back into text
    chunk_texts = tokenizer.batch_decode(
        chunk_token_ids, skip_special_tokens=True
    )

    return pd.DataFrame(
        {
            'summary': chunk_texts,
            'parentIndex': group_idx,
            'label': chunk_labels,
            'numericalLabel': chunk_numerical_labels,
        }
    )

def process_bills(bills: List[Bill], simplify: bool, chunk: bool, tokenizer = None) -> pd.DataFrame:
    """
    Given a list of Bill objects from the database, returns a dataframe with cleaned summaries and labels (both text and numerical).
    Args:
        bills: a list of `database.models.Bill` objects.
        simplify: if `True`, then bill policy areas will be binned into 8 possible classes (as opposed to the original 33).
        If `False`, then the policy areas are left as is. 
        chunk: if `True`, then the summaries will be tokenized and chunked into pieces that fit within the max token length of the model.
            If `False`, then the summaries are left as is.
        tokenizer: a HuggingFace tokenizer, required if `chunk` is `True`.
    Returns:
        A dataframe with columns 'summary', 'label', and 'numericalLabel'. 
    """

    if chunk and tokenizer is None:
        raise ValueError("tokenizer must be provided if chunk is True")

    summaries = [strip_html_tags(bill.summary) for bill in bills]
    labels = None
    if simplify:
        labels = [raw2simplified.get(bill.policy_area, None) for bill in bills]
    else:
        labels = [bill.policy_area for bill in bills]

    df = pd.DataFrame(list(zip(summaries, labels)), columns = ['summary', 'label'])
    df = df.dropna().reset_index(drop = True)
    
    if simplify:
        df['numericalLabel'] = simplified_encoder.transform(df['label'])
    else:
        df['numericalLabel'] = raw_encoder.transform(df['label'])
        
    if chunk:
        df = chunk_dataframe(df, tokenizer)
    
    return df

def make_collate_fn(tokenizer, indexed: bool = False, max_length: int = None):
    """
    Factory function that creates collate_fn to be passed to a torch `DataLoader` object. 
    Args:
        tokenizer: a HuggingFace tokenizer. This handles the work of padding sequences with zeroes
        indexed: if `True`, then the collate function will return indexed samples
        max_length: int passed to tokenizer to determine max token length of sequences. 
    """
    def collate_fn(batch: List[BillSample | IndexedBillSample]):
        x = tokenizer([item.x for item in batch],
                      padding = "longest", 
                      truncation = True,
                      max_length = max_length, 
                      return_tensors = "pt")
        y = torch.tensor([item.y for item in batch], dtype = torch.long)
        indices = None
        if indexed:
            indices = torch.tensor([item.parent_idx for item in batch], dtype = torch.long)

        # HF model expects labels in 'labels' parameter
        batch = x | {'labels': y}
        if indexed:
            batch = batch | {'parent_indices': indices}

        return batch

    return collate_fn

def get_dataloader(dataset: BillDataset, tokenizer, max_length: int = None, weighted_sampling: bool = False, indexed: bool = False, **kwargs) -> DataLoader:
    """
    Returns a torch `DataLoader` equipped with a collate_fn as returned by `make_collate_fn`.
    Args:
        dataset: a `BillDataset` object.
        tokenizer: a HuggingFace tokenizer passed to `make_collate_fn` to construct the collate_fn passed to the `DataLoader` constructor.
        max_length: int passed to tokenizer to determine max token length of sequences. 
        weighted: if `True`, then creates the `DataLoader` with a `WeightedRandomSampler` weighted by reciprocal of class counts
        indexed: if `True`, then creates a collate_fn that where chunks samples are indexed by which parent bill they belong to. 
    """

    sampler = None    
    if weighted_sampling:
        class_counts = dataset.y.value_counts().sort_index()
        class_weights = 1/class_counts
        sample_weights = dataset.y.map(class_weights).values
        sampler = WeightedRandomSampler(
            weights = sample_weights,
            num_samples = len(dataset),
            replacement = True
        )

    dataloader = DataLoader(
        dataset, 
        collate_fn = make_collate_fn(tokenizer, indexed, max_length),
        sampler = sampler,
        **kwargs
    )
    return dataloader

