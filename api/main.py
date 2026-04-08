from fastapi import FastAPI

from .utils import load_model
from .router import inference, info

app = FastAPI()

app.include_router(inference.router)
app.include_router(info.router)

@app.get("/")
async def root():
    return {"message": "hello world"}
