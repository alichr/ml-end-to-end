"""Streamlit frontend for Cat vs Dog Classifier."""

import os

import requests
import streamlit as st
from PIL import Image

API_URL = os.environ.get("API_URL", "http://localhost:8000")

# --- Page Config ---
st.set_page_config(
    page_title="PetVision AI — Cat vs Dog Classifier",
    page_icon="https://em-content.zobj.net/source/apple/391/paw-prints_1f43e.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* Global */
html, body, .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
.stApp {
    background: linear-gradient(160deg, #0f0c29 0%, #1a1a2e 40%, #16213e 100%);
}
.block-container {
    max-width: 1100px;
    padding-top: 2rem;
}

/* Hide default header/footer */
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

/* Hero */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
}
.hero h1 {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
    letter-spacing: -0.5px;
}
.hero p {
    color: #a0aec0;
    font-size: 1.15rem;
    font-weight: 400;
    margin-top: 0;
}

/* Cards */
.glass-card {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 2rem;
    backdrop-filter: blur(12px);
    margin-bottom: 1.5rem;
    transition: border-color 0.3s;
}
.glass-card:hover {
    border-color: rgba(102, 126, 234, 0.3);
}

/* Section headers */
.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    letter-spacing: 0.3px;
}

/* Result badge */
.result-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.8rem 1.6rem;
    border-radius: 50px;
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: 0.5px;
}
.result-cat {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    color: white;
    box-shadow: 0 4px 20px rgba(240, 147, 251, 0.3);
}
.result-dog {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    color: white;
    box-shadow: 0 4px 20px rgba(79, 172, 254, 0.3);
}

/* Confidence meter */
.conf-container {
    margin-top: 1.2rem;
}
.conf-row {
    display: flex;
    align-items: center;
    margin-bottom: 0.6rem;
    gap: 0.8rem;
}
.conf-label {
    width: 50px;
    font-size: 0.85rem;
    font-weight: 600;
    color: #cbd5e0;
    text-align: right;
}
.conf-bar-bg {
    flex: 1;
    height: 28px;
    background: rgba(255, 255, 255, 0.06);
    border-radius: 14px;
    overflow: hidden;
    position: relative;
}
.conf-bar-fill-cat {
    height: 100%;
    border-radius: 14px;
    background: linear-gradient(90deg, #f093fb, #f5576c);
    transition: width 0.6s ease;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    padding-right: 10px;
}
.conf-bar-fill-dog {
    height: 100%;
    border-radius: 14px;
    background: linear-gradient(90deg, #4facfe, #00f2fe);
    transition: width 0.6s ease;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    padding-right: 10px;
}
.conf-pct {
    font-size: 0.78rem;
    font-weight: 700;
    color: white;
    text-shadow: 0 1px 2px rgba(0,0,0,0.3);
}

/* Metadata pills */
.meta-row {
    display: flex;
    gap: 0.6rem;
    margin-top: 1rem;
    flex-wrap: wrap;
}
.meta-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.35rem 0.8rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 500;
    background: rgba(255, 255, 255, 0.06);
    color: #a0aec0;
    border: 1px solid rgba(255, 255, 255, 0.06);
}

/* Upload area styling */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(102, 126, 234, 0.3) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(102, 126, 234, 0.6) !important;
}

/* Button styling */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 2rem !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.3px !important;
    transition: all 0.3s !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5) !important;
}

/* Image container */
.img-frame {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.95);
    border-right: 1px solid rgba(255, 255, 255, 0.06);
}
.sidebar-status {
    padding: 1rem;
    border-radius: 12px;
    margin-bottom: 0.8rem;
}
.status-online {
    background: rgba(72, 187, 120, 0.1);
    border: 1px solid rgba(72, 187, 120, 0.25);
}
.status-offline {
    background: rgba(245, 101, 101, 0.1);
    border: 1px solid rgba(245, 101, 101, 0.25);
}
.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
    animation: pulse 2s infinite;
}
.dot-green { background: #48bb78; }
.dot-red { background: #f56565; }
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* Batch grid card */
.batch-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 12px;
    padding: 0.8rem;
    text-align: center;
    transition: transform 0.2s;
}
.batch-card:hover { transform: translateY(-2px); }
.batch-label {
    font-weight: 700;
    font-size: 0.9rem;
    margin-top: 0.4rem;
}
.batch-conf {
    font-size: 0.78rem;
    color: #a0aec0;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    color: #a0aec0;
    font-weight: 500;
    padding: 0.5rem 1.2rem;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border: none !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# --- Helpers ---
def check_health() -> dict[str, str | float | bool] | None:
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)  # type: ignore[no-untyped-call]
        if resp.status_code == 200:
            result: dict[str, str | float | bool] = resp.json()
            return result
    except requests.ConnectionError:
        pass
    return None


def render_confidence_bars(probs: dict[str, float]) -> str:
    cat_pct = probs.get("cat", 0) * 100
    dog_pct = probs.get("dog", 0) * 100
    return f"""
    <div class="conf-container">
        <div class="conf-row">
            <span class="conf-label">Cat</span>
            <div class="conf-bar-bg">
                <div class="conf-bar-fill-cat" style="width: {max(cat_pct, 2)}%;">
                    <span class="conf-pct">{cat_pct:.1f}%</span>
                </div>
            </div>
        </div>
        <div class="conf-row">
            <span class="conf-label">Dog</span>
            <div class="conf-bar-bg">
                <div class="conf-bar-fill-dog" style="width: {max(dog_pct, 2)}%;">
                    <span class="conf-pct">{dog_pct:.1f}%</span>
                </div>
            </div>
        </div>
    </div>
    """


def render_result_badge(pred: str, conf: float) -> str:
    icon = "\U0001f431" if pred == "cat" else "\U0001f436"
    cls = "result-cat" if pred == "cat" else "result-dog"
    return f"""
    <div style="text-align: center; margin: 1rem 0;">
        <span class="result-badge {cls}">{icon} {pred.upper()} &mdash; {conf * 100:.1f}%</span>
    </div>
    """


def render_meta_pills(result: dict) -> str:
    return f"""
    <div class="meta-row">
        <span class="meta-pill">\u23f1 {result['latency_ms']:.1f} ms</span>
        <span class="meta-pill">\U0001f4e6 Model v{result['model_version']}</span>
        <span class="meta-pill">\U0001f916 ONNX Runtime</span>
        <span class="meta-pill">\U0001f4f7 MobileNetV2</span>
    </div>
    """


# --- Sidebar ---
with st.sidebar:
    st.markdown(
        """
    <div style="text-align: center; padding: 1.5rem 0 1rem;">
        <div style="font-size: 2.2rem;">&#128062;</div>
        <div style="font-size: 1.1rem; font-weight: 700; color: #e2e8f0;
                    margin-top: 0.3rem;">PetVision AI</div>
        <div style="font-size: 0.75rem; color: #718096; margin-top: 0.15rem;">
            Intelligent Image Classification</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    health = check_health()
    if health and health["status"] == "healthy":
        uptime = float(health["uptime_seconds"])
        if uptime >= 3600:
            uptime_str = f"{uptime / 3600:.1f}h"
        elif uptime >= 60:
            uptime_str = f"{uptime / 60:.0f}m"
        else:
            uptime_str = f"{uptime:.0f}s"

        st.markdown(
            f"""
        <div class="sidebar-status status-online">
            <div style="font-size: 0.8rem; font-weight: 600; color: #48bb78;">
                <span class="status-dot dot-green"></span> System Online
            </div>
            <div style="font-size: 0.72rem; color: #a0aec0; margin-top: 0.4rem;">
                Model: v{health['model_version']}<br>
                Uptime: {uptime_str}<br>
                Engine: ONNX Runtime
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
        <div class="sidebar-status status-offline">
            <div style="font-size: 0.8rem; font-weight: 600; color: #f56565;">
                <span class="status-dot dot-red"></span> System Offline
            </div>
            <div style="font-size: 0.72rem; color: #a0aec0; margin-top: 0.4rem;">
                API: {API_URL}
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    st.markdown(
        """
    <div style="padding: 0.8rem; border-radius: 10px;
                background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06);">
        <div style="font-size: 0.78rem; font-weight: 600; color: #e2e8f0; margin-bottom: 0.5rem;">
            Model Specs</div>
        <div style="font-size: 0.72rem; color: #a0aec0; line-height: 1.7;">
            Architecture: MobileNetV2<br>
            Accuracy: 98.72%<br>
            Latency: ~4ms (CPU)<br>
            Size: 0.30 MB (ONNX)<br>
            Classes: Cat, Dog
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div style="position: fixed; bottom: 1rem; font-size: 0.65rem; color: #4a5568;">
        Built with FastAPI + ONNX Runtime + Streamlit
    </div>
    """,
        unsafe_allow_html=True,
    )


# --- Hero ---
st.markdown(
    """
<div class="hero">
    <h1>PetVision AI</h1>
    <p>State-of-the-art cat & dog classification powered by deep learning</p>
</div>
""",
    unsafe_allow_html=True,
)


# --- Main Content ---
tab_single, tab_batch = st.tabs(["\U0001f4f7  Single Image", "\U0001f5bc  Batch Upload"])


# --- Single Image Tab ---
with tab_single:
    st.markdown(
        """
    <div class="glass-card">
        <div class="section-title">\U0001f4e4 Upload an Image</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Drag and drop or browse — JPEG, PNG up to 5 MB",
        type=["jpg", "jpeg", "png"],
        key="single",
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        col_img, col_result = st.columns([1, 1], gap="large")

        with col_img:
            image = Image.open(uploaded_file)
            st.markdown('<div class="img-frame">', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            w, h = image.size
            st.markdown(
                f'<div style="text-align:center; font-size:0.72rem; color:#718096; '
                f'margin-top:0.3rem;">{uploaded_file.name} &middot; '
                f"{w}&times;{h} &middot; "
                f"{len(uploaded_file.getvalue()) / 1024:.0f} KB</div>",
                unsafe_allow_html=True,
            )

        with col_result:
            classify_btn = st.button(
                "\U0001f50d  Classify Image", key="classify_single", use_container_width=True
            )

            if classify_btn:
                with st.spinner(""):
                    uploaded_file.seek(0)
                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type,
                        )
                    }
                    try:
                        resp = requests.post(f"{API_URL}/predict", files=files, timeout=30)
                        if resp.status_code == 200:
                            result = resp.json()
                            st.markdown(
                                render_result_badge(
                                    result["predicted_class"], result["confidence"]
                                ),
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                render_confidence_bars(result["probabilities"]),
                                unsafe_allow_html=True,
                            )
                            st.markdown(render_meta_pills(result), unsafe_allow_html=True)
                        else:
                            error = resp.json()
                            st.error(f"**Error:** {error.get('detail', resp.text)}")
                    except requests.ConnectionError:
                        st.error(
                            f"Cannot connect to API at `{API_URL}`. "
                            "Make sure the server is running."
                        )


# --- Batch Tab ---
with tab_batch:
    st.markdown(
        """
    <div class="glass-card">
        <div class="section-title">\U0001f5bc Batch Classification</div>
        <p style="font-size: 0.85rem; color: #a0aec0; margin: 0;">
            Upload up to 16 images and classify them all at once.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    batch_files = st.file_uploader(
        "Drop multiple images here",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="batch",
        label_visibility="collapsed",
    )

    if batch_files:
        st.markdown(
            f'<div style="font-size: 0.85rem; color: #a0aec0; margin-bottom: 1rem;">'
            f"{len(batch_files)} image{'s' if len(batch_files) != 1 else ''} selected</div>",
            unsafe_allow_html=True,
        )

        if st.button(
            f"\U0001f680  Classify All ({len(batch_files)})",
            key="classify_batch",
            use_container_width=True,
        ):
            with st.spinner(f"Classifying {len(batch_files)} images..."):
                batch_payload = [
                    ("files", (f.name, f.getvalue(), f.type or "image/jpeg")) for f in batch_files
                ]
                try:
                    resp = requests.post(
                        f"{API_URL}/predict/batch", files=batch_payload, timeout=60
                    )
                    if resp.status_code == 200:
                        data = resp.json()

                        st.markdown(
                            f'<div style="font-size: 0.8rem; color: #718096; '
                            f'margin-bottom: 1rem;">'
                            f"Completed in {data['total_latency_ms']:.1f} ms "
                            f"({data['total_latency_ms'] / len(batch_files):.1f} ms/image)"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                        n_cols = min(4, len(batch_files))
                        cols = st.columns(n_cols, gap="medium")
                        for i, (f, pred) in enumerate(zip(batch_files, data["predictions"])):
                            with cols[i % n_cols]:
                                img = Image.open(f)
                                cls = pred["predicted_class"]
                                conf = pred["confidence"]
                                icon = "\U0001f431" if cls == "cat" else "\U0001f436"
                                color = "#f093fb" if cls == "cat" else "#4facfe"

                                st.markdown(
                                    '<div class="batch-card">',
                                    unsafe_allow_html=True,
                                )
                                st.image(img, use_container_width=True)
                                st.markdown(
                                    f'<div class="batch-label" style="color: {color};">'
                                    f"{icon} {cls.upper()}</div>"
                                    f'<div class="batch-conf">{conf * 100:.1f}%</div>'
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )
                    else:
                        error = resp.json()
                        st.error(f"**Error:** {error.get('detail', resp.text)}")
                except requests.ConnectionError:
                    st.error(
                        f"Cannot connect to API at `{API_URL}`. " "Make sure the server is running."
                    )
