---
title: Vision Suite
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

<div align="center">

# Vision Suite - CV Inference Platform

### A multi-task computer-vision inference platform — six vision tasks, one clean interface, CPU-ready.

Image Captioning | Object detection | Instance segmentation | Pose estimation | Multi-object tracking & counting | Image classification 

<br/>

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![YOLO26](https://img.shields.io/badge/YOLO26-Ultralytics-0B23A9)
![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-CPU-005CED?logo=onnx&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
![HF Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-FFD21E)

</div>

---

##  Overview

**Vision Suite** turns six computer-vision tasks into a single, deploy-ready web app. Upload an
image or video, pick a task, tune the thresholds, and get an annotated result with a live
detections breakdown and the good news - all running on **CPU**, no GPU required.

> _Have a Look:_ https://huggingface.co/spaces/Mystique03/vision-suite


<!-- ───────────────────────────────────────────────────────────── -->
<!-- TODO: replace the placeholders below with real screenshots / GIFs -->

##  Demo


<div align="center">

![UI overview](docs/samples/ui.png)

*Interface — sidebar controls, live preview, results card*

<br/>

| | |
|:--:|:--:|
| ![Detection](docs/samples/detection.gif) | ![Segmentation](docs/samples/segment.gif) |
| **Detection** — boxes + confidence list | **Segmentation** — instance masks |
| ![Pose](docs/samples/pose.jpg) | ![Captioning](docs/samples/caption.png) |
| **Pose** — keypoint skeletons | **Captioning** — BLIP scene description |

</div>


<!-- ───────────────────────────────────────────────────────────── -->

##  Features

| Task | Model | Input | Output |
|------|-------|:-----:|--------|
| **Object detection** | YOLO26n  | image / video | boxes + labels + confidence |
| **Instance segmentation** | YOLO26n-seg  | image / video | per-instance masks |
| **Pose estimation** | YOLO26n-pose  | image / video | keypoint skeletons |
| **Multi-object tracking** | YOLO26n + BoT-SORT | video | persistent IDs + **per-class unique counts** |
| **Image classification** | EfficientNet-B0 | image | top-5 ImageNet predictions |
| **Image captioning** | BLIP | image | natural-language description |

##  Highlights

- **CPU-first performance** — all YOLO models exported to **ONNX Runtime**  for fast inference without a GPU.
- **Browser-ready video** — OpenCV `mp4v` output is transcoded to **H.264** so results play inline in any browser.
- **Tunable inference** — confidence & IoU thresholds are exposed in the UI and threaded
  end-to-end into the models.
- **Observability** — every inference is logged to **Weights & Biases** (latency, confidence stats).
- **One-container deploy** — FastAPI + Streamlit supervised in a single Docker image, weights pre-baked at build time for instant cold starts.

## 📊 Performance

CPU inference latency, YOLO26n detection, single image, 30 warm runs averaged
(Intel Core i5-1135G7, no GPU):

| Model + runtime | Latency (ms/frame) |
|---|:--:|
| YOLOv8n · PyTorch | 215.6 |
| YOLOv8n · ONNX Runtime | 177.7 |
| YOLO26n · PyTorch | 227.8 |
| **YOLO26n · ONNX Runtime** | **145.8** |

Exporting to **ONNX Runtime cuts latency ~36%** (228 → 146 ms) via operator fusion and
static-shape graphs — the model's speed advantage shows up at the runtime level, making
real-time multi-task vision viable on commodity CPUs.


##  Tech Stack

**Inference:** Ultralytics YOLO26, ONNX Runtime, BLIP (Hugging Face Transformers), EfficientNet (torchvision)
**Serving:** FastAPI, Uvicorn
**Interface:** Streamlit
**Ops:** Docker, supervisord, Weights & Biases

## ⚡ Getting Started

### Local

```bash
# install deps
pip install -r requirements.txt

# terminal 1 — API
uvicorn app.main:app --port 8000

# terminal 2 — UI
streamlit run ui/streamlit_app.py
```


### Docker

```bash
docker build -t vision-suite .
docker run -p 7860:7860 vision-suite
```

Open <http://localhost:7860>.


##  Project Structure

```
vision-suite/
├── app/
│   ├── main.py                 # FastAPI app
│   ├── routers/inference.py    # /infer endpoint, dispatch, video transcode
│   └── services/
│       ├── model_loader.py     
│       └── monitor.py          # W&B logging
├── models/                     # 6 models
├── ui/
│   ├── streamlit_app.py        # UI
│   └── samples/                # demo image + video
├── Dockerfile                  # one-container build, weights pre-baked
├── supervisord.conf            # runs API + UI together
└── requirements.txt
```

##  Future Work

- Add Multimodal features
- Real-time webcam inference - coming soon


---

<div align="center">
Built by <a href="https://smithasreddy.vercel.app">Smitha S Reddy</a>
</div>
