from ultralytics import YOLO
import cv2 as cv
import torch
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

_weights = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")
_pt = os.path.join(_weights, "yolo26n-pose.pt")
_onnx = os.path.splitext(_pt)[0] + ".onnx"
if not os.path.exists(_onnx):
    YOLO(_pt).export(format="onnx", imgsz=640, simplify=True)
model = YOLO(_onnx, task="pose")

KPT_CONF = 0.5  # keypoints below this confidence are not drawn

def _gate_keypoints(result):
    """Zero out low-confidence keypoints in place so plot() skips them."""
    kp = result.keypoints
    if kp is None or kp.data is None or kp.data.numel() == 0:
        return
    data = kp.data.clone()  # (n_person, n_kpt, 3): x, y, conf; clone -> inference tensor is read-only
    data[data[..., 2] < KPT_CONF] = 0
    kp.data = data

def detect_pose_image(image, conf=0.25, iou=0.7):
    persons = []
    results = model.predict(image, conf=conf, iou=iou, verbose=False)
    r = results[0]

    if r.boxes is not None:
        for box_conf in r.boxes.conf.cpu().numpy():
            persons.append({"label": "person", "confidence": round(float(box_conf), 2)})

    _gate_keypoints(r)
    annotated_frame = r.plot(boxes=False)
    return annotated_frame, persons

def detect_pose_video(video, output_path='output.mp4', conf=0.25, iou=0.7):
    cap = cv.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video.")

    fps = cap.get(cv.CAP_PROP_FPS)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(output_path, fourcc, fps, (width, height))

    detections = []

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model.predict(frame, conf=conf, iou=iou, verbose=False)
            r = results[0]
            if r.boxes is not None:
                for box_conf in r.boxes.conf.cpu().numpy():
                    detections.append({"label": "person", "confidence": round(float(box_conf), 2)})
            _gate_keypoints(r)
            annotated_frame = r.plot(boxes=False)
            out.write(annotated_frame)

    finally:
        cap.release()
        out.release()

    return output_path, detections