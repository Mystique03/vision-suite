
from ultralytics import YOLO
import cv2 as cv
import torch
import os
from collections import defaultdict

device = "cuda" if torch.cuda.is_available() else "cpu"


def _draw_counts(frame, seen):
    """Overlay running unique-count per class, top-left."""
    y = 30
    for label in sorted(seen):
        text = f"{label}: {len(seen[label])}"
        cv.putText(frame, text, (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(frame, text, (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)
        y += 32

_weights = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")
_pt = os.path.join(_weights, "yolo26n.pt")
_onnx = os.path.splitext(_pt)[0] + ".onnx"
if not os.path.exists(_onnx):
    YOLO(_pt).export(format="onnx", imgsz=640, simplify=True)
model = YOLO(_onnx, task="detect")

def track_video(video_path, output_path, conf=0.4, iou=0.7):
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv.CAP_PROP_FPS)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(output_path, fourcc, fps, (width, height))

    detections = []
    seen = defaultdict(set)  # label -> set of unique track_ids

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.track(frame, tracker="botsort.yaml", persist=True, conf=conf, iou=iou, verbose=False)

            if results[0].boxes.id is None:
                _draw_counts(frame, seen)
                out.write(frame)
                continue

            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()

            for box, id, conf, cls in zip(boxes, track_ids, confs, classes):
                label = model.names[int(cls)]
                seen[label].add(int(id))
                detections.append({
                    "label": label,
                    "confidence": round(float(conf), 2),
                    "box": [int(x) for x in box],
                    "track_id": int(id)
                })

            annotated_frame = results[0].plot()
            _draw_counts(annotated_frame, seen)
            out.write(annotated_frame)
    finally:
        cap.release()
        out.release()

    return output_path, detections



        