---
title: CV Platform
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

<div align="center">

# 🔍 CV Platform

### A multi-task computer-vision inference platform — six vision tasks, one clean interface, CPU-ready.

Object detection · Instance segmentation · Pose estimation · Multi-object tracking · Image classification · Image captioning

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

## ✨ Overview

**CV Platform** turns six computer-vision tasks into a single, deploy-ready web app. Upload an
image or video, pick a task, tune the thresholds, and get an annotated result with a live
detections breakdown — all running on commodity **CPU**, no GPU required.

Built around **YOLO26** exported to **ONNX Runtime** for fast CPU inference, with a **BLIP**
vision-language model for natural-language image captioning.

<!-- ───────────────────────────────────────────────────────────── -->
<!-- TODO: replace the placeholders below with real screenshots / GIFs -->

## 🎬 Demo

<div align="center">

| | |
|:--:|:--:|
| ![UI overview](docs/samples/ui.png) | ![Detection](docs/samples/detection.png) |
| **Interface** — sidebar controls, live preview, results | **Detection** — boxes + confidence list |
| ![Segmentation](docs/samples/segmentation.png) | ![Pose](docs/samples/pose.png) |
| **Segmentation** — instance masks | **Pose** — keypoint skeletons |
| ![Tracking](docs/samples/tracking.gif) | ![Captioning](docs/samples/caption.png) |
| **Tracking** — IDs + unique object counts | **Captioning** — BLIP scene description |

</div>

> _Live demo:_ **[🤗 Hugging Face Space](https://huggingface.co/spaces/<your-username>/cv-platform)** _(coming soon)_

<!-- ───────────────────────────────────────────────────────────── -->

## 🧩 Features

| Task | Model | Input | Output |
|------|-------|:-----:|--------|
| **Object detection** | YOLO26n (ONNX) | image / video | boxes + labels + confidence |
| **Instance segmentation** | YOLO26n-seg (ONNX) | image / video | per-instance masks |
| **Pose estimation** | YOLO26n-pose (ONNX) | image / video | 17-keypoint skeletons (confidence-gated) |
| **Multi-object tracking** | YOLO26n + BoT-SORT | video | persistent IDs + **per-class unique counts** |
| **Image classification** | EfficientNet-B0 | image | top-5 ImageNet predictions |
| **Image captioning** | BLIP | image | natural-language description |

## 🚀 Highlights

- **CPU-first performance** — all YOLO models exported to **ONNX Runtime** (operator fusion,
  static-shape graphs) for fast inference without a GPU.
- **Memory-efficient model loading** — a single-model-in-memory loader hot-swaps models on
  demand and releases the previous one, keeping RAM bounded across all six tasks (fits free-tier hosting).
- **Browser-ready video** — OpenCV `mp4v` output is transcoded to **H.264** so results play
  inline in any browser.
- **Tunable inference** — confidence & IoU thresholds are exposed in the UI and threaded
  end-to-end into the models.
- **Observability** — every inference is logged to **Weights & Biases** (latency, confidence stats).
- **One-container deploy** — FastAPI + Streamlit supervised in a single Docker image, weights
  pre-baked at build time for instant cold starts.

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

## 🏗️ Architecture

```
                       ┌─────────────────────────────────────────────┐
   Browser  ──────────▶│  Streamlit UI            (port 7860, public) │
                       │    sidebar controls · preview · results card │
                       └───────────────────────┬─────────────────────┘
                                                │  HTTP (localhost)
                       ┌────────────────────────▼─────────────────────┐
                       │  FastAPI  /infer         (port 8000, internal)│
                       │    ├─ model_loader   one model in RAM, swapped│
                       │    ├─ models/*       YOLO26·ONNX, EfficientNet, BLIP
                       │    ├─ _to_h264       browser-playable video   │
                       │    └─ monitor        W&B per-inference logging│
                       └───────────────────────────────────────────────┘
                            both processes managed by supervisord
```

The UI is the only public port; it calls the FastAPI backend over `localhost` inside the
same container. Metadata (detections + latency) is returned via an `X-Meta` response header,
keeping media payloads clean.

## 🛠️ Tech Stack

**Inference:** Ultralytics YOLO26 · ONNX Runtime · BLIP (Hugging Face Transformers) · EfficientNet (torchvision)
**Serving:** FastAPI · Uvicorn
**Interface:** Streamlit
**Ops:** Docker · supervisord · Weights & Biases

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

Open <http://localhost:8501>.

### Docker

```bash
docker build -t cv-platform .
docker run -p 7860:7860 cv-platform
```

Open <http://localhost:7860>. The build **pre-bakes** all model weights into the image, so
the first request is instant.

## ☁️ Deployment (Hugging Face Spaces)

This repo is Space-ready: the README front-matter sets `sdk: docker` and `app_port: 7860`.
Push to a Docker Space and it builds and serves automatically — no extra config.

## 📁 Project Structure

```
cv-platform/
├── app/
│   ├── main.py                 # FastAPI app
│   ├── routers/inference.py    # /infer endpoint, dispatch, video transcode
│   └── services/
│       ├── model_loader.py     # single-model-in-memory hot-swap
│       └── monitor.py          # W&B logging
├── models/                     # one module per task (detection, pose, segmenter, tracker, classifier, captioner)
├── ui/
│   ├── streamlit_app.py        # UI
│   └── samples/                # bundled demo image + video
├── Dockerfile                  # one-container build, weights pre-baked
├── supervisord.conf            # runs API + UI together
└── requirements.txt
```

## 🗺️ Roadmap

- [ ] Real-time webcam inference (`streamlit-webrtc`)
- [ ] High-resolution video downscaling for faster throughput
- [ ] Zero-shot open-vocabulary detection (YOLO-World)
- [ ] Promptable segmentation (SAM)

---

<div align="center">
Built by <a href="https://smithasreddy.vercel.app">Smitha Reddy</a>
</div>
