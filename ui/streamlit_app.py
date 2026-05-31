import streamlit as st
import requests
import json
import hashlib
from pathlib import Path

API = "http://localhost:8000/infer"
SAMPLES = Path(__file__).parent / "samples"
FEATURES = ["classification", "caption", "detection", "segmentation", "tracking", "pose"]
IMAGE_ONLY = {"classification", "caption"}
VIDEO_ONLY = {"tracking"}

st.set_page_config(layout="wide", page_title="CV Platform", page_icon="🔍")

st.markdown(
    """
    <style>
    .block-container { padding-top: 2.5rem; max-width: 1300px; }
    .card-title { font-size: 1.25rem; font-weight: 700; margin-bottom: .25rem; }
    .card-sub { color: #6b7280; font-size: .85rem; margin-bottom: 1rem; }
    .badge { background:#eef2ff; color:#4338ca; padding:3px 10px; border-radius:999px;
             font-size:.75rem; font-weight:600; float:right; }
    .det-list { margin-top:.5rem; }
    .det-row { display:flex; align-items:center; padding:9px 12px; border-radius:8px; }
    .det-row:nth-child(odd) { background:#f6f7f9; }
    .dot { width:11px; height:11px; border-radius:50%; margin-right:11px; flex-shrink:0; }
    .lbl { flex:1; font-weight:500; color:#111827; }
    .val { color:#4b5563; font-variant-numeric:tabular-nums; font-weight:600; }
    .empty { color:#9ca3af; font-style:italic; padding:12px; }
    .caption-text { font-size:1.15rem; line-height:1.5; color:#111827; font-weight:500;
                    padding:14px 16px; margin-top:10px; background:#f6f7f9; border-radius:10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

ss = st.session_state
ss.setdefault("file_bytes", None)
ss.setdefault("file_name", None)
ss.setdefault("file_type", None)
ss.setdefault("result", None)


def set_input(data, name, ftype):
    ss.file_bytes, ss.file_name, ss.file_type = data, name, ftype
    ss.result = None


def color_for(label):
    h = int(hashlib.md5(label.encode()).hexdigest(), 16) % 360
    return f"hsl({h}, 65%, 50%)"


def render_items(items):
    if not items:
        return '<div class="empty">No objects detected.</div>'
    rows = ""
    for it in items:
        c = color_for(it["label"])
        if "count" in it:
            right = f'&times;{it["count"]}'
        else:
            right = f'{it["confidence"] * 100:.1f}%'
        rows += (
            f'<div class="det-row"><span class="dot" style="background:{c}"></span>'
            f'<span class="lbl">{it["label"]}</span><span class="val">{right}</span></div>'
        )
    return f'<div class="det-list">{rows}</div>'


st.title("🔍 CV Platform")
st.caption("YOLO26 inference — detection, segmentation, pose, tracking, classification & captioning")

# ---------------- SIDEBAR: CONTROLS ----------------
with st.sidebar:
    st.header("Controls")
    feature = st.selectbox("Feature", FEATURES)

    up = st.file_uploader("Upload (1 file)", ["png", "jpg", "jpeg", "mp4"],
                          accept_multiple_files=False)
    if up is not None:
        set_input(up.getvalue(), up.name, up.type)

    conf = st.slider("Confidence", 0.0, 1.0, 0.25, 0.05,
                     help="Minimum confidence for detections.")
    iou = st.slider("IoU", 0.0, 1.0, 0.7, 0.05,
                    help="IoU threshold for Non-Maximum Suppression.")
    if feature in IMAGE_ONLY:
        st.caption(f"ℹ️ Confidence/IoU don't apply to {feature}.")

left, right = st.columns(2, gap="large")

# ---------------- INPUT CARD ----------------
with left:
    with st.container(border=True):
        st.markdown('<div class="card-title">Input</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-sub">Pick a sample or upload in the sidebar, then Run.</div>',
                    unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        if c1.button("🖼️ Sample image", use_container_width=True):
            set_input((SAMPLES / "sample.jpg").read_bytes(), "sample.jpg", "image/jpeg")
        if c2.button("🎬 Sample video", use_container_width=True):
            set_input((SAMPLES / "sample.mp4").read_bytes(), "sample.mp4", "video/mp4")

        # preview
        if ss.file_bytes:
            if ss.file_type and ss.file_type.startswith("video"):
                st.video(ss.file_bytes)
            else:
                st.image(ss.file_bytes, use_container_width=True)
        else:
            st.markdown('<div class="empty">No file selected.</div>', unsafe_allow_html=True)

        run = st.button("Run", type="primary", use_container_width=True)

# ---------------- RUN ----------------
if run:
    if not ss.file_bytes:
        ss.result = {"error": "Upload a file or pick a sample first."}
    elif feature in VIDEO_ONLY and not (ss.file_type or "").startswith("video"):
        ss.result = {"error": "Tracking needs a video file."}
    elif feature in IMAGE_ONLY and (ss.file_type or "").startswith("video"):
        ss.result = {"error": "Classification works on images only."}
    else:
        with right:
            with st.spinner("Running inference… (video may take a while)"):
                try:
                    resp = requests.post(
                        API,
                        params={"feature": feature, "conf": conf, "iou": iou},
                        files={"file": (ss.file_name, ss.file_bytes, ss.file_type)},
                        timeout=600,
                    )
                except requests.RequestException as e:
                    ss.result = {"error": f"Request failed: {e}"}
                    resp = None
                if resp is not None:
                    if not resp.ok:
                        try:
                            detail = resp.json().get("detail", "Inference failed.")
                        except Exception:
                            detail = "Inference failed."
                        ss.result = {"error": detail}
                    elif feature == "caption":
                        body = resp.json()
                        ss.result = {"kind": "caption", "text": body["caption"],
                                     "media": ss.file_bytes, "latency": body["latency_ms"]}
                    elif feature == "classification":
                        body = resp.json()
                        ss.result = {"kind": "classification", "items": body["items"],
                                     "latency": body["latency_ms"]}
                    else:
                        meta = json.loads(resp.headers.get("X-Meta", "{}"))
                        is_video = (ss.file_type or "").startswith("video")
                        ss.result = {
                            "kind": "video" if is_video else "image",
                            "media": resp.content,
                            "items": meta.get("items", []),
                            "latency": meta.get("latency_ms"),
                        }

# ---------------- RESULT CARD ----------------
with right:
    with st.container(border=True):
        res = ss.result
        latency = res.get("latency") if res else None
        badge = f'<span class="badge">{latency} ms</span>' if latency is not None else ""
        st.markdown(f'<div class="card-title">Result {badge}</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-sub">Annotated output and detections.</div>',
                    unsafe_allow_html=True)

        if not res:
            st.markdown('<div class="empty">Run an analysis to see results.</div>',
                        unsafe_allow_html=True)
        elif "error" in res:
            st.error(res["error"])
        elif res["kind"] == "caption":
            st.image(res["media"], use_container_width=True)
            st.markdown(f'<div class="caption-text">“{res["text"]}”</div>',
                        unsafe_allow_html=True)
        elif res["kind"] == "classification":
            st.markdown(render_items(res["items"]), unsafe_allow_html=True)
        else:
            if res["kind"] == "image":
                st.image(res["media"], use_container_width=True)
                st.download_button("⬇️ Download image", res["media"],
                                   file_name="result.jpg", mime="image/jpeg",
                                   use_container_width=True)
            elif res["kind"] == "video":
                st.video(res["media"])
                st.download_button("⬇️ Download video", res["media"],
                                   file_name="result.mp4", mime="video/mp4",
                                   use_container_width=True)
            st.markdown(render_items(res["items"]), unsafe_allow_html=True)
