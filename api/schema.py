from typing import List, Literal, Dict
from pydantic import BaseModel

class InfRequest(BaseModel):
    """
    Schema for user request for model inference
    """
    text: str
    top_k: int = 1
    mode: Literal["macro", "chunk"]     # specifies whether to apply model to get a macro-level label, chunk-level labels, or both
    return_chunk_text: bool = False

class InfResponseBase(BaseModel):
    """
    Base schema for all responses to model inference request. 
    Stores request metadata such as number of tokens and chunks
    """
    num_tokens: int
    num_chunks: int
class MacroInfResponse(InfResponseBase):
    """
    Schema for response to model inference request in macro mode. 
    """
    prediction: str
    probabilities: Dict[str, float]

class ChunkResponseItem(BaseModel):
    """
    Schema for chunk-level data returned when calling model inference in chunk mode.
    """
    text: str | None = None
    prediction: str
    probabilities: Dict[str, float]

class ChunkInfResponse(InfResponseBase):
    """
    Schema for response to model inference request in chunk mode
    """
    chunks: List[ChunkResponseItem]
