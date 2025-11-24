import os
import hmac
import json
from typing import Optional
import torch
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

#configurations
APP_NAME = "Bias & Fairness Detection API"
MODEL_PATH = os.getenv("MODEL_PATH", "./model")
SECRET_STRING = (os.getenv("SECRET_STRING") or "").strip()

#inference settings
MAX_LEN = int(os.getenv("MAX_LEN", "256"))  #match training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#app
app = FastAPI(title=APP_NAME)

class DetectionRequest(BaseModel):
    response_text: str


#load model, tokenizer, and calibration
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()
except Exception as e:
    raise RuntimeError(f"Failed to load model/tokenizer from {MODEL_PATH}: {e}")

#load calibration and thresholds saved by train.py
CAL_FILE = os.path.join(MODEL_PATH, "calibration_and_thresholds.json")
T = 1.0
THRESHOLD = 0.5
#attempt to read calibration file
try:
    with open(CAL_FILE, "r", encoding="utf-8") as f:
        cal = json.load(f)
    #prefer tuned threshold; fall back to deploy_threshold or 0.5
    T = float(cal.get("temperature", 1.0))
    THRESHOLD = float(
        cal.get(
            "global_threshold_tuned",
            cal.get("deploy_threshold", 0.5)
        )
    )
except FileNotFoundError:
    #safe fallbacks if calibration file isn't present
    T, THRESHOLD = 1.0, 0.5
except Exception as e:
    #if calibration file is corrupted, then default safely
    print(f"[WARN] Failed to read calibration file: {e}")
    T, THRESHOLD = 1.0, 0.5

#require API key for access
def _require_api_key(request: Request):
    if not SECRET_STRING:
        raise HTTPException(status_code=500, detail="Server missing SECRET_STRING")
    client_key = (request.headers.get("x-api-key", "") or "").strip()
    if not hmac.compare_digest(client_key, SECRET_STRING):
        raise HTTPException(status_code=401, detail="Unauthorized")

#prediction probability function
def _predict_proba(text: str) -> float:
    """Return calibrated P(biased) for a single text."""
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
    )
    enc = {k: v.to(DEVICE, non_blocking=True) for k, v in enc.items()}
    with torch.inference_mode():
        logits = model(**enc).logits
        #apply temperature from training-time calibration
        logits = logits / float(T)
        probs = torch.softmax(logits, dim=-1)
        p_biased = float(probs[0, 1].item())
    return p_biased


#routes
@app.get("/")
# health check
def root():
    return {
        "message": f"{APP_NAME} is running!",
        "device": str(DEVICE),
        "temperature": T,
        "threshold": THRESHOLD,
    }

#bias detection endpoint
@app.post("/detect")
# detect bias in provided text
def detect_bias(payload: DetectionRequest, request: Request):
    _require_api_key(request)
    text = payload.response_text or ""
    if not text.strip():
        raise HTTPException(status_code=400, detail="response_text must be non-empty")

    p_biased = _predict_proba(text)
    pred = p_biased >= THRESHOLD
    #construct response
    return {
        "input_text": text,
        "bias_detected": bool(pred),
        "p_biased": p_biased,
        "threshold": THRESHOLD,
        "temperature": T,
        "confidence_scores": {
            "neutral": 1.0 - p_biased,
            "biased": p_biased,
        },
    }