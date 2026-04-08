from fastapi import FastAPI, Response, status, HTTPException, Depends, APIRouter

from ..model import macro_inference, chunk_inference
from ..schema import InfRequest, MacroInfResponse, ChunkInfResponse

router = APIRouter(
    prefix = "/inference",
    tags = ['Inference']
)

@router.post("/", response_model = MacroInfResponse | ChunkInfResponse)
async def inference(request: InfRequest):
    mode = request.mode
    if mode == "macro":
        response = macro_inference(
            text = request.text, 
            top_k = request.top_k
        )
        return response
    elif mode == "chunk":
        response = chunk_inference(
            text = request.text,
            top_k = request.top_k,
            return_chunk_text = request.return_chunk_text
        )
        return response
