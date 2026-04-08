from fastapi import APIRouter

from ..utils import possible_raw_labels

router = APIRouter(
    prefix = "/info",
    tags = ['Info']
)

@router.get("/")
async def get_info():
    summary = (
        "A fine-tuned DistilRoBERTa text classifier for predicting the "
        "policy area of U.S. congressional bills, trained on bill summaries and " 
        "their associated policy area labels sourced from api.congress.gov. For "
        "texts exceeding 512 tokens, the model applies a sliding window with "
        "overlap and aggregates chunk-level logits before predicting. Returns "
        "the top-k most likely policy areas with associated probabilities."
    )

    response = {
        "labels": possible_raw_labels,
        "base_model": "distilbert/distilroberta-base",
        "tokenizer": "RobertaTokenizer",
        "summary": summary
    }
    return response