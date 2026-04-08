from fastapi import FastAPI
import torch

from .utils import load_model, chunk_and_tokenize, raw_encoder
from .schema import InfRequest, InfResponse

app = FastAPI()

load = load_model()
tokenizer = load["tokenizer"]
model = load["model"]
model.eval()

@app.get("/")
async def root():
    return {"message": "hello world"}

@app.post("/inference", response_model = InfResponse)
async def inference(user_input: InfRequest):
    chunked_input = chunk_and_tokenize(user_input.text, tokenizer)
    chunks = chunked_input["chunks"]
    out = model(input_ids = chunks)
    
    full_logits = out.logits.sum(dim = 0)
    full_probs = full_logits.softmax(dim = -1)
    
    top_probs, top_labels = torch.topk(full_probs, k = user_input.top_k)
    top_labels_decoded = raw_encoder.inverse_transform(top_labels)
    other_prob = 1 - top_probs.sum()
    
    prob_response = {label.item(): probs.item() for label, probs in 
                     zip(top_labels_decoded, top_probs)}
    if other_prob > 1e-3:
        prob_response = prob_response | {"Other": other_prob}

    return {"prediction": top_labels_decoded[0].item(),
            "probabilities": prob_response,
            "num_tokens": chunked_input["num_tokens"],
            "num_chunks": chunks.shape[0]}