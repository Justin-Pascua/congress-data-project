import torch
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import re

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

def load_model():
    path = "./api/model/"
    model = AutoModelForSequenceClassification.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    return {"tokenizer": tokenizer,
            "model": model}

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

# this is different from version in ml/utils/data.py because it keeps the input tokenized,
# and returns as an (N, L) tensor, where N is the number of chunks and L is the max seq len
def chunk_and_tokenize(text: str, tokenizer, overlap: int = 50):
    """
    Tokenizes text and splits into chunks that fit within the tokenizer's max sequence length.
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

    token_ids = tokenizer.encode(text, add_special_tokens = False, return_tensors = "pt")

    if token_ids.shape[-1] <= effective_max:
        return {"chunks": token_ids,
                "num_tokens": len(token_ids[-1])}

    chunks = []
    step = effective_max - overlap
    start = 0

    while start < token_ids.shape[-1]:
        end = start + effective_max
        chunks.append(token_ids[:, start:end])
        start += step

    # last two chunks may be short, so pad if needed
    # second to last may be short because of overlap
    for i in [-1, -2]:
        if chunks[i].shape[-1] < effective_max and len(chunks) > 1:
            chunks[i] = torch.cat([
                chunks[i], 
                torch.full(size = (1, effective_max - chunks[i].shape[-1], ), fill_value =tokenizer.pad_token_id)
                ], dim = 1)

    return {"chunks": torch.cat(chunks, dim = 0),
            "num_tokens": len(token_ids[-1]),
            "num_chunks": len(chunks)}
