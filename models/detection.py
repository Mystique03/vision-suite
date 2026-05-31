from ultralytics import YOLO
import torch
import cv2 as cv
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

_weights = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")
_pt = os.path.join(_weights, 'yolo26n.pt')
_onnx = os.path.splitext(_pt)[0] + '.onnx'
if not os.path.exists(_onnx):
    YOLO(_pt).export(format='onnx', imgsz=640, simplify=True)
model = YOLO(_onnx, task='detect')

def detect(image, conf=0.25, iou=0.7):
    output = model.predict(image, conf=conf, iou=iou, verbose=False)

    results = output[0]
    labels = model.names

    detections = []

    if len(output[0].boxes) == 0: # no detections
        return results.plot(), []

    for box, conf, cls in zip(
        results.boxes.xyxy.cpu().numpy(),
        results.boxes.conf.cpu().numpy(),
        results.boxes.cls.cpu().numpy()
    ):
        detections.append({
            "label": labels[int(cls)],
            "confidence": round(float(conf), 2),
            "box": [int(x) for x in box]
        })
    annotated_frame = results.plot()

    return annotated_frame, detections


def detect_video(video_path, output_path, conf=0.25, iou=0.7):
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

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

            results = model.predict(frame, conf=conf, iou=iou, verbose=False)[0]

            if len(results.boxes) == 0:
                out.write(frame)
                continue

            for box, conf, cls in zip(
                results.boxes.xyxy.cpu().numpy(),
                results.boxes.conf.cpu().numpy(),
                results.boxes.cls.cpu().numpy()
            ):
                detections.append({
                    "label": model.names[int(cls)],
                    "confidence": round(float(conf), 2),
                    "box": [int(x) for x in box]
                })

            out.write(results.plot())
    finally:
        cap.release()
        out.release()

    return output_path, detections


