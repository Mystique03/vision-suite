import os
import wandb
import numpy as np
from dotenv import load_dotenv

load_dotenv()  # load .env so wandb sees WANDB_API_KEY (uvicorn/streamlit don't auto-load it)

# Online when an API key is present; offline otherwise. Override with WANDB_MODE.
_WANDB_MODE = os.getenv("WANDB_MODE", "online" if os.getenv("WANDB_API_KEY") else "offline")

_wandb_initialized = False

def _ensure_wandb():
    global _wandb_initialized
    if not _wandb_initialized:
        wandb.init(project="vision-suite", mode=_WANDB_MODE)
        _wandb_initialized = True

def log_inference(feature, latency, input_metadata, confidence, error=None):
    if confidence:
        mean_conf = np.mean(confidence)
        min_conf = np.min(confidence)
    else:
        mean_conf = None
        min_conf = None

    log_data = {
        "feature": str(feature),
        "latency": float(latency),
        **input_metadata,
        "mean_confidence": mean_conf,
        "min_confidence": min_conf,
        "error": error
    }
    _ensure_wandb()
    wandb.log(log_data)