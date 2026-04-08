from typing import List, Optional, Literal, Tuple, Dict
from pydantic import BaseModel
from datetime import datetime

class InfRequest(BaseModel):
    text: str
    top_k: int = 1

class InfResponse(BaseModel):
    prediction: str
    # dict with class labels as keys and probabilities as values
    probabilities: Dict[str, float]
    num_tokens: int
    num_chunks: int
