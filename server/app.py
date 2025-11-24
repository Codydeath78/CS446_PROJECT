from fastapi import FastAPI
from pydantic import BaseModel
import os
from pathlib import Path
from google.cloud import storage
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

#initialize FastAPI app
app = FastAPI()

#load model from GCS
GCS_MODEL_PATH = os.environ.get(
    "GCS_MODEL_PATH",
    "gs://cs446project-474923-safety-data/models/safety-detector-v1",
)
#local path to store the downloaded model
LOCAL_MODEL_DIR = Path("/tmp/model")

class TextPayload(BaseModel):
    text: str

model = None
tokenizer = None

#function to download model files from GCS to local directory
def download_model():
    #if we already downloaded once in this container, then don't redo it
    if LOCAL_MODEL_DIR.exists() and any(LOCAL_MODEL_DIR.iterdir()):
        return
    #create local model directory if it doesn't exist
    LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    #parse GCS_MODEL_PATH to get bucket name and prefix
    if not GCS_MODEL_PATH.startswith("gs://"):
        raise ValueError(f"GCS_MODEL_PATH must start with gs://, got: {GCS_MODEL_PATH}")

    #strip "gs://" and split into bucket and prefix
    path = GCS_MODEL_PATH[5:]  #remove "gs://"
    parts = path.split("/", 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""

    print(f"[startup] Downloading model from bucket={bucket_name}, prefix={prefix!r}")

    #initialize GCS client and get bucket
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    #list all blobs under the prefix
    blobs = list(bucket.list_blobs(prefix=prefix))
    if not blobs:
        raise RuntimeError(f"No blobs found under gs://{bucket_name}/{prefix}")
    #download each blob to the local model directory
    for blob in blobs:
        #blob.name is like "models/safety-detector-v1/config.json"
        rel_path = blob.name[len(prefix):].lstrip("/") if prefix else blob.name
        dest_path = LOCAL_MODEL_DIR / rel_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[startup] Downloading {blob.name} -> {dest_path}")
        blob.download_to_filename(dest_path)

#load model and tokenizer at startup
@app.on_event("startup")
def load_model():
    global model, tokenizer
    download_model()
    #load tokenizer and model from local directory
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_DIR)
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")

#health check endpoint
@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Safety detector is running and ready for commands.",
        "predict_endpoint": "/predict",
    }

#prediction endpoint
@app.post("/predict")
def predict(payload: TextPayload):
    text = payload.text or ""
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt",
    )
    #move inputs to GPU if available
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    #perform inference
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1)[0].cpu().numpy()

    #assuming label 1 = unsafe, 0 = safe
    unsafe_prob = float(probs[1])
    safe_prob = float(probs[0])
    #return probabilities
    return {
        "unsafe": unsafe_prob,
        "safe": safe_prob,
    }