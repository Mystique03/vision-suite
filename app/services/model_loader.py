import importlib
import sys
import gc
import torch

MODEL_REGISTRY = {
    "classification": "models.classifier",
    "caption":        "models.captioner",
    "detection":      "models.detection",
    "pose":           "models.pose",
    "tracking":       "models.tracker",
    "segmentation":   "models.segmenter",
}

_current_feature = None
_current_module = None

def load_model(feature):
    global _current_feature, _current_module

    if feature not in MODEL_REGISTRY:
        raise ValueError("Model not found.")

    if feature == _current_feature:
        return _current_module

    if _current_feature is not None:
        sys.modules.pop(MODEL_REGISTRY[_current_feature], None)
        _current_module = None
        gc.collect()
        torch.cuda.empty_cache()

    _current_module = importlib.import_module(MODEL_REGISTRY[feature])
    _current_feature = feature
    return _current_module

    

