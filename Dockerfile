FROM python:3.11-slim

WORKDIR /cv-platform

# System libraries OpenCV needs (the slim image doesn't ship them)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install -r requirements.txt && pip install supervisor

# Pre-bake model weights into the image: download (BLIP, EfficientNet) and export
# (YOLO26 .onnx) at BUILD time so the first request isn't blocked on downloads.
RUN python -c "import models.classifier, models.captioner, models.detection, models.segmenter, models.pose, models.tracker"

EXPOSE 7860

CMD supervisord -c supervisord.conf
