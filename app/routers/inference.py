from fastapi import APIRouter, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, Response
from app.services.model_loader import load_model
from app.services.monitor import log_inference

import numpy as np
import tempfile
import os
import time
import json
import subprocess
from collections import Counter

import cv2 as cv
import imageio_ffmpeg

router = APIRouter()

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _items_image(result):
    """Per-object [{label, confidence}] from (frame, detections)."""
    return [{"label": d["label"], "confidence": d["confidence"]} for d in result[1]]

def _items_classification(result):
    """Top-k [{label, confidence}] from list of (label, score)."""
    return [{"label": l, "confidence": round(float(s), 3)} for l, s in result]

def _items_counts(result, unique=False):
    """Per-class counts from video detections. unique=True dedupes by track_id (tracking)."""
    dets = result[1]
    if unique:
        seen = {}
        for d in dets:
            seen.setdefault(d["label"], set()).add(d.get("track_id"))
        return [{"label": k, "count": len(v)} for k, v in sorted(seen.items())]
    counts = Counter(d["label"] for d in dets)
    return [{"label": k, "count": n} for k, n in sorted(counts.items())]

def _to_h264(src_path: str) -> str:
    """Transcode OpenCV mp4v output to browser-playable H.264. Returns new path."""
    fd, dst_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", src_path,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            dst_path,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return dst_path

def _media_type(file: UploadFile) -> str:
    content_type = file.content_type or ""
    if content_type.startswith("image") or content_type.startswith("video"):
        return content_type.split("/")[0]
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    return content_type

@router.post("/infer")
async def run(feature: str, file: UploadFile, background_tasks: BackgroundTasks,
              conf: float = 0.25, iou: float = 0.7):

    # (method_name, conf_fn -> confidences for monitor, items_fn -> UI list)
    IMAGE_DISPATCH = {
        "classification": ("predict",            lambda r: [s for _, s in r],          _items_classification),
        "caption":        ("caption",            lambda r: None,                       lambda r: []),
        "detection":      ("detect",             lambda r: [d["confidence"] for d in r[1]], _items_image),
        "segmentation":   ("segment_image",      lambda r: [d["confidence"] for d in r[1]], _items_image),
        "pose":           ("detect_pose_image",  lambda r: [d["confidence"] for d in r[1]], _items_image),
    }

    VIDEO_DISPATCH = {
        "detection":    ("detect_video",      lambda r: [d["confidence"] for d in r[1]], lambda r: _items_counts(r)),
        "segmentation": ("segment_video",     lambda r: [d["confidence"] for d in r[1]], lambda r: _items_counts(r)),
        "pose":         ("detect_pose_video",  lambda r: [d["confidence"] for d in r[1]], lambda r: _items_counts(r)),
        "tracking":     ("track_video",       lambda r: [d["confidence"] for d in r[1]], lambda r: _items_counts(r, unique=True)),
    }

    conf = float(conf)  # ensure plain Python float (ultralytics rejects numpy float32)
    iou = float(iou)

    content = await file.read()

    input_metadata = {
        "filename": file.filename,
        "content_type": file.content_type,
    }

    media = _media_type(file)
    start = time.time()

    if media == "image":
        if feature not in IMAGE_DISPATCH:
            raise HTTPException(status_code=400, detail=f"{feature} not supported for images")
        model = load_model(feature)
        method_name, conf_fn, items_fn = IMAGE_DISPATCH[feature]
        narr = np.frombuffer(content, np.uint8)
        image = cv.imdecode(narr, cv.IMREAD_COLOR)
        result = getattr(model, method_name)(image, conf=conf, iou=iou)
        confidence = conf_fn(result)
        items = items_fn(result)

    elif media == "video":
        if feature not in VIDEO_DISPATCH:
            raise HTTPException(status_code=400, detail=f"{feature} not supported for videos")
        model = load_model(feature)
        method_name, conf_fn, items_fn = VIDEO_DISPATCH[feature]
        in_fd, input_path = tempfile.mkstemp(suffix=".mp4")
        out_fd, output_path = tempfile.mkstemp(suffix=".mp4")
        os.close(in_fd)
        os.close(out_fd)
        try:
            with open(input_path, "wb") as f:
                f.write(content)
            result = getattr(model, method_name)(input_path, output_path, conf=conf, iou=iou)
            confidence = conf_fn(result)
            items = items_fn(result)
        finally:
            os.unlink(input_path)

    else:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    total_time = time.time() - start
    log_inference(feature, total_time, input_metadata, confidence)

    meta = json.dumps({"items": items, "latency_ms": round(total_time * 1000)})

    if media == "video":
        playable_path = _to_h264(output_path)
        background_tasks.add_task(os.unlink, output_path)
        background_tasks.add_task(os.unlink, playable_path)
        return FileResponse(playable_path, media_type="video/mp4", headers={"X-Meta": meta})

    if feature == "caption":
        return {"caption": result, "latency_ms": round(total_time * 1000)}

    if feature == "classification":
        return {"items": items, "latency_ms": round(total_time * 1000)}

    frame = result[0]
    _, buf = cv.imencode(".jpg", frame)
    return Response(content=buf.tobytes(), media_type="image/jpeg", headers={"X-Meta": meta})



