
from ultralytics import YOLO, SAM
import torch
import cv2 as cv
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

_weights = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")
sam_model = SAM(os.path.join(_weights, "mobile_sam.pt"))
sam_model.to(device)

_yolo_pt = os.path.join(_weights, 'yolo26n-seg.pt')
_yolo_onnx = os.path.splitext(_yolo_pt)[0] + '.onnx'
if not os.path.exists(_yolo_onnx):
    YOLO(_yolo_pt).export(format='onnx', imgsz=640, simplify=True)
yolo_model = YOLO(_yolo_onnx, task='segment')

def _boxes_to_detections(result):
    dets = []
    if result.boxes is None:
        return dets
    for conf, cls in zip(result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy()):
        dets.append({"label": yolo_model.names[int(cls)], "confidence": round(float(conf), 2)})
    return dets

def segment_image(image, conf=0.25, iou=0.7):
    output = yolo_model.predict(image, conf=conf, iou=iou, verbose=False)
    detections = _boxes_to_detections(output[0])
    return output[0].plot(boxes=False, labels=False), detections

def segment_video(video, output_path='output.mp4', conf=0.25, iou=0.7):
    cap = cv.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")

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

            output = yolo_model.predict(frame, conf=conf, iou=iou, verbose=False)

            if len(output[0].boxes) == 0:
                out.write(frame)
                continue

            detections.extend(_boxes_to_detections(output[0]))
            annotated_frame = output[0].plot(boxes=False, labels=False)
            out.write(annotated_frame)
    finally:
        cap.release()
        out.release()

    return output_path, detections
    
