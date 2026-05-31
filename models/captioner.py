"""BLIP image captioning. Weights auto-download from HF Hub on first load, cached locally."""
import torch
import cv2 as cv
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"

_MODEL = "Salesforce/blip-image-captioning-base"
processor = BlipProcessor.from_pretrained(_MODEL)
model = BlipForConditionalGeneration.from_pretrained(_MODEL).to(device)
model.eval()

def caption(image, conf=None, iou=None):  # conf/iou unused: captioning has no detections
    rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)  # cv2 decodes BGR; BLIP expects RGB
    pil = Image.fromarray(rgb)
    inputs = processor(pil, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=40)
    return processor.decode(out[0], skip_special_tokens=True)
