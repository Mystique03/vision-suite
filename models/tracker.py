
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

MIN_FRAMES = 10  # an ID must persist this many frames to be counted (filters blips / ID switches)

def track_video(video_path, output_path, conf=0.4, iou=0.7):
    conf, iou = float(conf), float(iou)  # ultralytics rejects numpy float32

    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv.CAP_PROP_FPS)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(output_path, fourcc, fps, (width, height))

    detections = []
    id_frames = defaultdict(int)       # track_id -> frames seen
    id_label = {}                      # track_id -> label
    confirmed = defaultdict(set)       # label -> set of confirmed track_ids (>= MIN_FRAMES)
    first = True  # frame 0 starts a fresh tracker (resets IDs/state per video)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.track(frame, tracker="botsort.yaml", persist=not first, conf=conf, iou=iou, verbose=False)
            first = False

            if results[0].boxes.id is None:
                _draw_counts(frame, confirmed)
                out.write(frame)
                continue

            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()

            for box, track_id, box_conf, cls in zip(boxes, track_ids, confs, classes):
                track_id = int(track_id)
                label = model.names[int(cls)]
                id_label[track_id] = label
                id_frames[track_id] += 1
                if id_frames[track_id] == MIN_FRAMES:   # crossed persistence threshold -> count it
                    confirmed[label].add(track_id)
                detections.append({
                    "label": label,
                    "confidence": round(float(box_conf), 2),
                    "box": [int(x) for x in box],
                    "track_id": track_id
                })

            annotated_frame = results[0].plot()
            _draw_counts(annotated_frame, confirmed)
            out.write(annotated_frame)
    finally:
        cap.release()
        out.release()

    # keep only detections from confirmed IDs so downstream counts match the overlay
    confirmed_ids = {tid for ids in confirmed.values() for tid in ids}
    detections = [d for d in detections if d["track_id"] in confirmed_ids]

    return output_path, detections



        