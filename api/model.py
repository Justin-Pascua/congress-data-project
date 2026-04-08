import torch
from .utils import load_model, chunk_and_tokenize, raw_encoder

load = load_model()
tokenizer = load["tokenizer"]
model = load["model"]
model.eval()

def macro_inference(text: str, top_k: int = 1):
    """
    Applies model to a text input. The text is tokenized and divided into chunks fitting 
    into the model's context window. The model is applied to each chunk, and the model 
    outputs are aggregated to infer labels for the entire text.
    Args:
        text: the text to be classified by the model
        top_k: the number of labels to assign to the input
    """
    chunked_input = chunk_and_tokenize(text, tokenizer)
    chunks = chunked_input["chunks"]
    num_chunks = chunked_input["num_chunks"]
    num_tokens = chunked_input["num_tokens"]
    
    out = model(input_ids = chunks)

    full_logits = out.logits.sum(dim = 0)
    full_probs = torch.round(full_logits.softmax(dim = -1), decimals = 3)
    
    top_probs, top_labels = torch.topk(full_probs, k = top_k)
    top_labels_decoded = raw_encoder.inverse_transform(top_labels)
    other_prob = 1 - top_probs.sum()
    
    prob_response = {label.item(): probs.item() for label, probs in 
                     zip(top_labels_decoded, top_probs)}
    if other_prob > 1e-3:
        prob_response = prob_response | {"Other": other_prob.item()}

    return {"prediction": top_labels_decoded[0].item(),
            "probabilities": prob_response,
            "num_chunks": num_chunks,
            "num_tokens": num_tokens}

def chunk_inference(text, top_k: int = 1, return_chunk_text: bool = False):
    """
    Applies model to a text input. The text is tokenized and divided into chunks fitting 
    into the model's context window. The model is applied to each chunk, and labels are
    assigned to each individual chunk.
    Args:
        text: the text to be classified by the model
        top_k: the number of labels to assign to the input
        return_chunk_text: if True, returns each chunk decoded back to text
    """
    chunked_input = chunk_and_tokenize(text, tokenizer)
    chunks = chunked_input["chunks"]
    num_chunks = chunked_input["num_chunks"]
    num_tokens = chunked_input["num_tokens"]

    out = model(input_ids = chunks)
    
    chunk_probs = out.logits.softmax(dim = 1)
    chunk_top_probs, chunk_top_labels = torch.topk(chunk_probs, k = top_k, dim = 1)
    chunk_top_probs = torch.round(chunk_top_probs, decimals = 3)
    # need to flatten before feeding into encoder
    chunk_top_labels_decoded = (raw_encoder
                                .inverse_transform(chunk_top_labels.flatten())
                                .reshape(chunk_top_labels.shape))
    # combined prob of classes outisde of top_k
    other_probs = 1 - chunk_top_probs.sum(dim = 1, keepdim = True)

    chunk_response_items = []
    for chunk, probs, labels_decoded, other_prob in zip(
        chunks, chunk_top_probs, chunk_top_labels_decoded, other_probs
    ):
        prob_response = {label.item(): probs.item() for label, probs in 
                        zip(labels_decoded, probs)}
        if other_prob > 1e-3:
            prob_response = prob_response | {"Other": other_prob.item()}
        
        chunk_response_item = {"prediction": labels_decoded[0].item(),
                               "probabilities": prob_response}
        if return_chunk_text:
            chunk_response_item["text"] = tokenizer.decode(chunk, skip_special_tokens = True)

        chunk_response_items.append(chunk_response_item)

    return {"num_chunks": num_chunks,
            "num_tokens": num_tokens,
            "chunks": chunk_response_items}
