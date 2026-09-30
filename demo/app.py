from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FAVICON_PATH = Path(__file__).parent / "favicon.ico"

st.set_page_config(
    page_title="Steelix",
    page_icon=str(FAVICON_PATH) if FAVICON_PATH.exists() else "🔩",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #F0FDFF;
        color: #0F172A;
    }

    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #E2E8F0;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.5rem;
        padding-left: 1.25rem;
        padding-right: 1.25rem;
    }

    .sidebar-brand {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 6px;
        margin-bottom: 1.5rem;
        padding-bottom: 1.25rem;
        border-bottom: 1px solid #E2E8F0;
    }

    .sidebar-logo {
        width: 120px;
        height: 120px;
        object-fit: contain;
        display: block;
        margin-bottom: 2px;
    }

    .sidebar-brand-name {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.01em;
        line-height: 1.2;
    }

    .sidebar-brand-tag {
        font-size: 0.62rem;
        color: #64748B;
        font-weight: 400;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        text-align: center;
        margin-top: 0;
    }

    .sidebar-section-label {
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #06B6D4;
        margin-bottom: 0.5rem;
        margin-top: 1.25rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background-color: transparent;
        border-bottom: 1px solid #E2E8F0;
        padding: 0;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border: none;
        border-bottom: 2px solid transparent;
        color: #64748B;
        font-size: 0.825rem;
        font-weight: 500;
        padding: 0.75rem 1.25rem;
        margin-bottom: -1px;
        border-radius: 0;
        font-family: 'Inter', sans-serif;
        letter-spacing: 0;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: #0F172A;
        background-color: #ECFEFF;
    }

    .stTabs [aria-selected="true"] {
        background-color: transparent !important;
        border-bottom: 2px solid #06B6D4 !important;
        color: #0F172A !important;
        font-weight: 600 !important;
    }

    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 1.5rem;
        background-color: transparent !important;
    }

    .page-header {
        margin-bottom: 1.75rem;
    }

    .page-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.02em;
        line-height: 1.3;
        margin: 0 0 0.25rem 0;
    }

    .page-subtitle {
        font-size: 0.825rem;
        color: #64748B;
        font-weight: 400;
        margin: 0;
    }

    .metric-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1px;
        background-color: #E2E8F0;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        overflow: hidden;
        margin: 1.25rem 0;
    }

    .metric-cell {
        background-color: #ffffff;
        padding: 1rem 1.25rem;
    }

    .metric-label {
        font-size: 0.7rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: 0.35rem;
    }

    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0F172A;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -0.02em;
    }

    .metric-value.pass { color: #22C55E; }
    .metric-value.flagged { color: #EF4444; }
    .metric-value.review { color: #F59E0B; }

    .status-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 3px;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }

    .status-pass { background-color: rgba(34,197,94,0.1); color: #22C55E; border: 1px solid rgba(34,197,94,0.25); }
    .status-flagged { background-color: rgba(239,68,68,0.1); color: #EF4444; border: 1px solid rgba(239,68,68,0.25); }
    .status-review { background-color: rgba(245,158,11,0.1); color: #F59E0B; border: 1px solid rgba(245,158,11,0.25); }

    .upload-zone {
        border: 1.5px dashed #BAE6FD;
        border-radius: 6px;
        padding: 2rem;
        text-align: center;
        background-color: #F0FDFF;
        transition: border-color 0.15s;
    }

    .upload-zone:hover {
        border-color: #06B6D4;
    }

    .section-divider {
        border: none;
        border-top: 1px solid #E2E8F0;
        margin: 1.5rem 0;
    }

    .detail-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1px;
        background-color: #E2E8F0;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        overflow: hidden;
        margin-bottom: 1.25rem;
    }

    .detail-cell {
        background-color: #ffffff;
        padding: 0.65rem 1rem;
        display: flex;
        align-items: baseline;
        gap: 0.5rem;
    }

    .detail-key {
        font-size: 0.72rem;
        color: #64748B;
        font-weight: 500;
        min-width: 130px;
        flex-shrink: 0;
    }

    .detail-val {
        font-size: 0.8rem;
        color: #0F172A;
        font-family: 'JetBrains Mono', monospace;
    }

    .kpi-strip {
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        gap: 1px;
        background-color: #E2E8F0;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        overflow: hidden;
        margin-bottom: 1.5rem;
    }

    .kpi-cell {
        background-color: #ffffff;
        padding: 0.85rem 1rem;
        text-align: center;
    }

    .kpi-label {
        font-size: 0.65rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: 0.3rem;
    }

    .kpi-value {
        font-size: 1.3rem;
        font-weight: 700;
        color: #0F172A;
        font-family: 'JetBrains Mono', monospace;
    }

    .kpi-value.pass { color: #22C55E; }
    .kpi-value.flagged { color: #EF4444; }
    .kpi-value.review { color: #F59E0B; }

    .info-notice {
        background-color: #ECFEFF;
        border: 1px solid #BAE6FD;
        border-left: 3px solid #06B6D4;
        border-radius: 0 6px 6px 0;
        padding: 0.75rem 1rem;
        font-size: 0.8rem;
        color: #0F172A;
        margin: 0.75rem 0;
    }

    .warn-notice {
        background-color: #FFFBEB;
        border: 1px solid #FDE68A;
        border-left: 3px solid #F59E0B;
        border-radius: 0 6px 6px 0;
        padding: 0.75rem 1rem;
        font-size: 0.8rem;
        color: #0F172A;
        margin: 0.75rem 0;
    }

    .error-notice {
        background-color: #FEF2F2;
        border: 1px solid #FECACA;
        border-left: 3px solid #EF4444;
        border-radius: 0 6px 6px 0;
        padding: 0.75rem 1rem;
        font-size: 0.8rem;
        color: #0F172A;
        margin: 0.75rem 0;
    }

    .success-notice {
        background-color: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-left: 3px solid #22C55E;
        border-radius: 0 6px 6px 0;
        padding: 0.75rem 1rem;
        font-size: 0.8rem;
        color: #0F172A;
        margin: 0.75rem 0;
    }

    .stButton > button {
        background-color: #06B6D4;
        color: #ffffff;
        border: none;
        border-radius: 4px;
        font-size: 0.825rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        padding: 0.5rem 1.25rem;
        letter-spacing: 0;
        transition: background-color 0.15s;
        cursor: pointer;
    }

    .stButton > button:hover {
        background-color: #0891B2;
        border: none;
    }

    .stButton > button:focus {
        box-shadow: 0 0 0 2px rgba(6,182,212,0.3);
        outline: none;
    }

    .stButton > button[kind="secondary"] {
        background-color: #ffffff;
        color: #64748B;
        border: 1px solid #E2E8F0;
    }

    .stButton > button[kind="secondary"]:hover {
        background-color: #F0FDFF;
        color: #0F172A;
        border-color: #06B6D4;
    }

    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stNumberInput > div > div > input {
        background-color: #ffffff !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 4px !important;
        color: #0F172A !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.825rem !important;
    }

    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #06B6D4 !important;
        box-shadow: 0 0 0 2px rgba(6,182,212,0.15) !important;
    }

    .stTextInput > div > div > input::placeholder,
    .stTextArea > div > div > textarea::placeholder {
        color: #94A3B8 !important;
    }

    .stSelectbox > div > div,
    .stSelectbox > div > div > div,
    [data-baseweb="select"] > div,
    [data-baseweb="select"] > div > div {
        background-color: #ffffff !important;
        border-color: #E2E8F0 !important;
        color: #0F172A !important;
    }

    [data-baseweb="select"] [data-testid="stMarkdownContainer"] {
        color: #0F172A !important;
    }

    [data-baseweb="popover"] {
        background-color: #ffffff !important;
    }

    [data-baseweb="menu"] {
        background-color: #ffffff !important;
    }

    [data-baseweb="menu"] li {
        background-color: #ffffff !important;
        color: #0F172A !important;
    }

    [data-baseweb="menu"] li:hover {
        background-color: #ECFEFF !important;
    }

    [data-baseweb="option"]:hover {
        background-color: #ECFEFF !important;
    }

    [data-baseweb="tag"] {
        background-color: #E0F2FE !important;
        color: #0F172A !important;
    }

    [data-testid="stFileUploader"] {
        background-color: #ffffff !important;
        border-radius: 6px !important;
    }

    [data-testid="stFileUploader"] > div {
        background-color: #ffffff !important;
    }

    [data-testid="stFileUploader"] section {
        background-color: #ffffff !important;
        border: 1.5px dashed #BAE6FD !important;
        border-radius: 6px !important;
        transition: border-color 0.15s !important;
    }

    [data-testid="stFileUploader"] section:hover {
        border-color: #06B6D4 !important;
    }

    [data-testid="stFileUploader"] section > div {
        background-color: #ffffff !important;
    }

    [data-testid="stFileUploader"] button {
        background-color: #ffffff !important;
        color: #06B6D4 !important;
        border: 1px solid #BAE6FD !important;
        border-radius: 4px !important;
    }

    [data-testid="stFileUploader"] button:hover {
        border-color: #06B6D4 !important;
        background-color: #ECFEFF !important;
    }

    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploaderDropzoneInstructions"] {
        color: #64748B !important;
    }

    [data-baseweb="notification"] {
        background-color: #E0F2FE !important;
        color: #0F172A !important;
    }

    .stSlider > div[data-baseweb="slider"] > div > div > div {
        background-color: #06B6D4 !important;
    }

    [data-baseweb="slider"] [role="slider"] {
        background-color: #06B6D4 !important;
        border-color: #06B6D4 !important;
    }

    [data-baseweb="slider"] > div:first-child {
        background-color: #E0F2FE !important;
    }

    [data-baseweb="slider"] [data-baseweb="thumb"] {
        background-color: #06B6D4 !important;
        border-color: #06B6D4 !important;
    }

    label, .stSlider label, .stTextInput label, .stTextArea label,
    .stSelectbox label, .stNumberInput label, .stFileUploader label {
        font-size: 0.77rem !important;
        font-weight: 500 !important;
        color: #64748B !important;
        letter-spacing: 0 !important;
    }

    .stRadio > div {
        gap: 0.5rem;
    }

    .stRadio > div > label {
        background-color: #ffffff !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 4px !important;
        padding: 0.3rem 0.75rem !important;
        font-size: 0.8rem !important;
        color: #64748B !important;
        cursor: pointer;
        transition: all 0.1s;
    }

    .stRadio > div > label:has(input:checked) {
        border-color: #06B6D4 !important;
        color: #0F172A !important;
        background-color: #ECFEFF !important;
    }

    .stRadio [data-baseweb="radio"] span {
        background-color: #06B6D4 !important;
        border-color: #06B6D4 !important;
    }

    .stDataFrame {
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        overflow: hidden;
    }

    .stDataFrame table {
        font-size: 0.78rem;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.7rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B !important;
    }

    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.3rem !important;
        color: #0F172A !important;
    }

    .stMarkdown h4 {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }

    .stMarkdown h3 {
        font-size: 1rem;
        font-weight: 600;
        color: #0F172A;
        letter-spacing: -0.01em;
    }

    [data-testid="stExpander"] {
        border: 1px solid #E2E8F0 !important;
        border-radius: 6px !important;
        background-color: #ffffff !important;
    }

    [data-testid="stExpander"] > details > summary {
        background-color: #ffffff !important;
        color: #0F172A !important;
    }

    [data-testid="stExpanderDetails"] {
        background-color: #F0FDFF !important;
        border-top: 1px solid #E2E8F0;
    }

    .stSpinner > div {
        border-top-color: #06B6D4 !important;
    }

    .stAlert {
        border-radius: 4px;
        font-size: 0.8rem;
    }

    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #F0FDFF; }
    ::-webkit-scrollbar-thumb { background: #BAE6FD; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #06B6D4; }

    .stDeployButton, footer, #MainMenu { display: none !important; }

    header[data-testid="stHeader"] {
        background-color: #F0FDFF !important;
        border-bottom: 1px solid #E2E8F0;
    }

    .block-container {
        padding-top: 5rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1400px;
    }

    .hw-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1.25rem;
    }

    .hw-section {
        background-color: #ffffff;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
    }

    .hw-section-title {
        font-size: 0.68rem;
        font-weight: 700;
        color: #06B6D4;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.75rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #E0F2FE;
    }

    .hw-row {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        padding: 0.3rem 0;
        border-bottom: 1px solid rgba(226,232,240,0.8);
    }

    .hw-row:last-child { border-bottom: none; }

    .hw-key {
        font-size: 0.73rem;
        color: #64748B;
    }

    .hw-val {
        font-size: 0.73rem;
        font-family: 'JetBrains Mono', monospace;
        color: #0F172A;
    }

    .pkg-row {
        display: flex;
        justify-content: space-between;
        padding: 0.3rem 0;
        border-bottom: 1px solid rgba(226,232,240,0.8);
    }

    .pkg-row:last-child { border-bottom: none; }

    .pkg-name {
        font-size: 0.73rem;
        color: #64748B;
    }

    .pkg-ver {
        font-size: 0.73rem;
        font-family: 'JetBrains Mono', monospace;
        color: #06B6D4;
    }

    .empty-state {
        text-align: center;
        padding: 3rem 2rem;
        color: #64748B;
    }

    .empty-state-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #0F172A;
        margin-bottom: 0.5rem;
    }

    .empty-state-desc {
        font-size: 0.78rem;
        color: #64748B;
        line-height: 1.6;
    }

    .gpu-indicator {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        font-size: 0.73rem;
        padding: 0.2rem 0.5rem;
        border-radius: 3px;
    }

    .gpu-on {
        background-color: rgba(34,197,94,0.08);
        color: #22C55E;
        border: 1px solid rgba(34,197,94,0.2);
    }

    .gpu-off {
        background-color: #F0FDFF;
        color: #64748B;
        border: 1px solid #BAE6FD;
    }

    .col-label {
        font-size: 0.72rem;
        font-weight: 600;
        color: #06B6D4;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.75rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #E0F2FE;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    LOGO_PATH = Path(__file__).parent / "logo.png"
    if LOGO_PATH.exists():
        import base64
        with open(LOGO_PATH, "rb") as _f:
            _logo_b64 = base64.b64encode(_f.read()).decode()
        st.markdown(
            f"""
            <div class="sidebar-brand">
                <img class="sidebar-logo" src="data:image/png;base64,{_logo_b64}" alt="Steelix logo" />
                <div class="sidebar-brand-tag">Intelligent Steel Surface Defect Detection & Traceability AI System</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="sidebar-brand-name">Steelix</div>
                <div class="sidebar-brand-tag">Smart AI-Based Steel Defect Detection</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="sidebar-section-label">Model</div>', unsafe_allow_html=True)
    weights_input = st.text_input(
        "Weights path",
        value="models/best.pt",
        help="Path to trained YOLOv8 weights file.",
    )
    weights_path = ROOT / weights_input if not Path(weights_input).is_absolute() else Path(weights_input)

    conf_thresh = st.slider("Confidence threshold", 0.10, 0.95, 0.30, 0.05)
    iou_thresh = st.slider("IoU threshold (NMS)", 0.10, 0.95, 0.45, 0.05)
    imgsz = st.select_slider("Inference image size", [320, 416, 512, 640, 768, 1024], value=640)

    st.markdown('<div class="sidebar-section-label">Database</div>', unsafe_allow_html=True)
    db_path_input = st.text_input("Database path", value="storage/steelvision.db")
    db_path = ROOT / db_path_input if not Path(db_path_input).is_absolute() else Path(db_path_input)


tab_infer, tab_history, tab_analytics, tab_system = st.tabs(
    ["Inference", "Passport History", "Analytics", "System"]
)

with tab_infer:
    st.markdown(
        '<div class="page-header"><p class="page-title">Defect Inference</p>'
        '<p class="page-subtitle">Upload a steel surface image or video to detect and classify surface defects.</p></div>',
        unsafe_allow_html=True,
    )

    if not weights_path.exists():
        fallback = ROOT / "yolov8n.pt"
        if fallback.exists():
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(fallback, weights_path)
            st.markdown(
                "<div class='info-notice'>Using pretrained YOLOv8-nano weights (general object detection). "
                "For accurate steel defect results, train on the NEU-DET dataset:<br>"
                "<code>.venv\\Scripts\\activate &amp;&amp; python scripts/train.py --epochs 50 --batch 8 --device cpu</code>"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='error-notice'>Model weights not found at <code>{weights_path}</code>. "
                "No fallback yolov8n.pt found. Run training first: "
                "<code>python scripts/train.py --epochs 50 --batch 8</code></div>",
                unsafe_allow_html=True,
            )

    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown('<div class="col-label">Input</div>', unsafe_allow_html=True)
        mode = st.radio("Source type", ["Image", "Video"], horizontal=True, label_visibility="collapsed")

        uploaded = st.file_uploader(
            "Upload file",
            type=["jpg", "jpeg", "png", "bmp"] if mode == "Image" else ["mp4", "avi", "mov", "mkv"],
            help="Upload a steel surface image or video.",
            label_visibility="collapsed",
        )

        operator_notes = st.text_area(
            "Operator notes",
            height=80,
            placeholder="Optional notes for this inspection...",
        )
        run_btn = st.button("Run Inference", type="primary")

    with col_result:
        st.markdown('<div class="col-label">Results</div>', unsafe_allow_html=True)
        result_placeholder = st.empty()

        if run_btn and uploaded:
            if not weights_path.exists():
                st.markdown(
                    f"<div class='error-notice'>Model weights not found at <code>{weights_path}</code>.</div>",
                    unsafe_allow_html=True,
                )
            else:
                try:
                    from src.inference.predictor import UltralyticsPredictor
                    from src.passport.passport_manager import PassportManager

                    predictor = UltralyticsPredictor(
                        weights_path=weights_path,
                        conf=conf_thresh,
                        iou=iou_thresh,
                        imgsz=imgsz,
                    )
                    pm = PassportManager(db_path=db_path)

                    if mode == "Image":
                        with tempfile.NamedTemporaryFile(suffix=Path(uploaded.name).suffix, delete=False) as tmp:
                            tmp.write(uploaded.read())
                        tmp_path = Path(tmp.name)

                        with st.spinner("Running inference..."):
                            inspection_id = pm.start_inspection(
                                input_source="upload",
                                model_name=weights_path.name,
                                backend="PyTorch",
                                conf_threshold=conf_thresh,
                                iou_threshold=iou_thresh,
                                operator_notes=operator_notes or None,
                            )
                            result = predictor.predict_image(tmp_path)

                        import cv2
                        import numpy as np
                        annotated_rgb = cv2.cvtColor(result["annotated_image"], cv2.COLOR_BGR2RGB)
                        st.image(annotated_rgb, caption="Annotated output", width="stretch")

                        status = result["inspection"]
                        status_cls = "pass" if status == "PASS" else ("flagged" if "FLAG" in status else "review")
                        fps_val = f"{1000/result['latency_ms']:.1f}" if result["latency_ms"] > 0 else "—"

                        st.markdown(
                            f"""
                            <div class="metric-row">
                                <div class="metric-cell">
                                    <div class="metric-label">Detections</div>
                                    <div class="metric-value">{result['num_detections']}</div>
                                </div>
                                <div class="metric-cell">
                                    <div class="metric-label">Latency</div>
                                    <div class="metric-value">{result['latency_ms']:.1f}<span style="font-size:0.75rem;color:#64748b;"> ms</span></div>
                                </div>
                                <div class="metric-cell">
                                    <div class="metric-label">FPS</div>
                                    <div class="metric-value">{fps_val}</div>
                                </div>
                                <div class="metric-cell">
                                    <div class="metric-label">Status</div>
                                    <div class="metric-value {status_cls}">{status}</div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        if result["detections"]:
                            st.markdown('<div class="col-label" style="margin-top:1rem;">Detected Defects</div>', unsafe_allow_html=True)
                            import pandas as pd
                            df = pd.DataFrame(result["detections"])
                            df["confidence"] = df["confidence"].apply(lambda x: f"{x:.3f}")
                            df["bbox_xyxy"] = df["bbox_xyxy"].apply(
                                lambda b: f"[{b[0]:.0f}, {b[1]:.0f}, {b[2]:.0f}, {b[3]:.0f}]"
                            )
                            st.dataframe(df[["class_name", "confidence", "bbox_xyxy"]], width="stretch")
                        else:
                            st.markdown(
                                "<div class='success-notice'>No defects detected above the confidence threshold.</div>",
                                unsafe_allow_html=True,
                            )

                        from src.analytics.defect_analytics import DefectAnalytics
                        analytics = DefectAnalytics()
                        for det in result["detections"]:
                            h, w = result["annotated_image"].shape[:2]
                            x1, y1, x2, y2 = det["bbox_xyxy"]
                            cx_norm = ((x1 + x2) / 2) / w
                            cy_norm = ((y1 + y2) / 2) / h
                            det["cx_norm"] = cx_norm
                            det["cy_norm"] = cy_norm
                            det["region"] = "Center"
                            det["bbox_area_px2"] = (x2 - x1) * (y2 - y1)
                            det["frame_percentage"] = det["bbox_area_px2"] / (w * h) * 100
                            analytics.add_detection(det)

                        summary = analytics.get_summary()
                        summary["total_frame_detections"] = result["num_detections"]
                        summary["unique_tracked_defects"] = result["num_detections"]
                        summary["avg_confidence"] = float(
                            sum(d["confidence"] if isinstance(d["confidence"], float)
                                else float(d["confidence"]) for d in result["detections"])
                            / max(1, result["num_detections"])
                        )
                        summary["class_counts"] = {
                            d["class_name"]: sum(
                                1 for x in result["detections"] if x["class_name"] == d["class_name"]
                            )
                            for d in result["detections"]
                        }
                        summary["region_counts"] = {}
                        summary["low_conf_count"] = sum(
                            1 for d in result["detections"]
                            if (d["confidence"] if isinstance(d["confidence"], float) else float(d["confidence"])) < 0.35
                        )
                        summary["pattern_count"] = 0
                        summary["longest_recurring_sequence"] = 0

                        pm.finalize_inspection(inspection_id, summary)
                        st.markdown(
                            f"<div class='success-notice'>Passport saved — ID: <code>{inspection_id}</code></div>",
                            unsafe_allow_html=True,
                        )

                    else:
                        with tempfile.NamedTemporaryFile(suffix=Path(uploaded.name).suffix, delete=False) as tmp:
                            tmp.write(uploaded.read())
                        tmp_path = Path(tmp.name)

                        out_path = ROOT / "results" / "predictions" / ("annotated_" + uploaded.name)
                        out_path.parent.mkdir(parents=True, exist_ok=True)

                        with st.spinner("Processing video..."):
                            video_result = predictor.predict_video(tmp_path, output_path=out_path)

                        st.markdown(
                            f"<div class='success-notice'>Video processed — avg latency {video_result['avg_latency_ms']:.1f} ms/frame.</div>",
                            unsafe_allow_html=True,
                        )
                        if out_path.exists():
                            st.markdown(
                                f"<div class='info-notice'>Annotated video saved to <code>{out_path}</code></div>",
                                unsafe_allow_html=True,
                            )

                except ImportError as e:
                    st.markdown(
                        f"<div class='error-notice'>Missing dependency: {e}<br>Run: <code>pip install -r requirements.txt</code></div>",
                        unsafe_allow_html=True,
                    )
                except Exception as e:
                    st.exception(e)

        elif run_btn and not uploaded:
            st.markdown(
                "<div class='warn-notice'>Upload a file before running inference.</div>",
                unsafe_allow_html=True,
            )
        elif not run_btn:
            st.markdown(
                """
                <div class="empty-state">
                    <div class="empty-state-title">No results yet</div>
                    <div class="empty-state-desc">Upload an image or video and click Run Inference to analyze steel surface defects.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


with tab_history:
    st.markdown(
        '<div class="page-header"><p class="page-title">Passport History</p>'
        '<p class="page-subtitle">Browse and search past inspection records stored in the digital steel passport database.</p></div>',
        unsafe_allow_html=True,
    )

    try:
        from src.passport.passport_manager import PassportManager
        import pandas as pd

        pm = PassportManager(db_path=db_path)
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            status_filter = st.selectbox("Status filter", ["All", "PASS", "REVIEW", "FLAGGED"])
        with col_f2:
            limit = st.number_input("Records to show", min_value=5, max_value=500, value=50)
        with col_f3:
            search_q = st.text_input("Search", placeholder="Batch ID, source, or notes...")

        kwargs: dict = {"limit": int(limit)}
        if status_filter != "All":
            kwargs["status_filter"] = status_filter

        if search_q:
            records = pm.search(search_q)
        else:
            records = pm.list_inspections(**kwargs)

        if not records:
            st.markdown(
                """
                <div class="empty-state">
                    <div class="empty-state-title">No records found</div>
                    <div class="empty-state-desc">The database has no inspection records matching the current filters.<br>Run an inference first to create a passport entry.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            df = pd.DataFrame(records)

            rename = {
                "inspection_id": "Inspection ID",
                "batch_id": "Batch ID",
                "start_time": "Start Time",
                "inspection_status": "Status",
                "total_frame_detections": "Detections",
                "unique_tracked_defects": "Unique Defects",
                "avg_confidence": "Avg Conf",
                "most_frequent_class": "Top Class",
                "most_affected_region": "Top Region",
                "input_source": "Source",
                "model_name": "Model",
                "duration_seconds": "Duration (s)",
            }
            display_cols = [c for c in rename if c in df.columns]
            df_display = df[display_cols].rename(columns=rename)

            def _colour_status(val):
                colours = {"PASS": "#22c55e", "REVIEW": "#f59e0b", "FLAGGED": "#ef4444"}
                c = colours.get(val, "#94a3b8")
                return f"color: {c}; font-weight: 600"

            st.dataframe(
                df_display.style.map(_colour_status, subset=["Status"] if "Status" in df_display.columns else []),
                width="stretch",
                height=320,
            )

            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            st.markdown('<div class="col-label">Inspection Detail</div>', unsafe_allow_html=True)

            selected_id = st.selectbox(
                "Select inspection",
                options=[r["inspection_id"] for r in records],
                label_visibility="collapsed",
            )
            if selected_id:
                passport = pm.get_passport(selected_id)
                if passport:
                    status_val = passport.get("inspection_status", "")
                    status_cls_map = {"PASS": "pass", "FLAGGED": "flagged", "REVIEW": "review"}
                    s_cls = status_cls_map.get(status_val, "")

                    st.markdown(
                        f"""
                        <div class="detail-grid">
                            <div class="detail-cell"><span class="detail-key">Inspection ID</span><span class="detail-val">{passport.get('inspection_id', '—')}</span></div>
                            <div class="detail-cell"><span class="detail-key">Batch ID</span><span class="detail-val">{passport.get('batch_id', '—')}</span></div>
                            <div class="detail-cell"><span class="detail-key">Status</span><span class="detail-val"><span class="status-badge status-{s_cls}">{status_val}</span></span></div>
                            <div class="detail-cell"><span class="detail-key">Reason</span><span class="detail-val">{passport.get('status_reason', '—')}</span></div>
                            <div class="detail-cell"><span class="detail-key">Start</span><span class="detail-val">{passport.get('start_time', '—')}</span></div>
                            <div class="detail-cell"><span class="detail-key">End</span><span class="detail-val">{passport.get('end_time', '—')}</span></div>
                            <div class="detail-cell"><span class="detail-key">Duration</span><span class="detail-val">{passport.get('duration_seconds', 0):.1f} s</span></div>
                            <div class="detail-cell"><span class="detail-key">Total detections</span><span class="detail-val">{passport.get('total_frame_detections', 0)}</span></div>
                            <div class="detail-cell"><span class="detail-key">Unique defects</span><span class="detail-val">{passport.get('unique_tracked_defects', 0)}</span></div>
                            <div class="detail-cell"><span class="detail-key">Avg confidence</span><span class="detail-val">{passport.get('avg_confidence', 0):.3f}</span></div>
                            <div class="detail-cell"><span class="detail-key">Top class</span><span class="detail-val">{passport.get('most_frequent_class', '—')}</span></div>
                            <div class="detail-cell"><span class="detail-key">Top region</span><span class="detail-val">{passport.get('most_affected_region', '—')}</span></div>
                            <div class="detail-cell"><span class="detail-key">Model</span><span class="detail-val">{passport.get('model_name', '—')}</span></div>
                            <div class="detail-cell"><span class="detail-key">Backend</span><span class="detail-val">{passport.get('backend', '—')}</span></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    cc_json = passport.get("class_counts_json", "{}")
                    try:
                        cc = json.loads(cc_json) if isinstance(cc_json, str) else cc_json
                    except Exception:
                        cc = {}
                    if cc:
                        import plotly.graph_objects as go
                        fig = go.Figure(go.Bar(
                            x=list(cc.keys()),
                            y=list(cc.values()),
                            marker_color="#06B6D4",
                            marker_line_width=0,
                        ))
                        fig.update_layout(
                            title=None,
                            template="plotly_white",
                            height=240,
                            margin=dict(l=0, r=0, t=16, b=0),
                            plot_bgcolor="#ffffff",
                            paper_bgcolor="#ffffff",
                            font=dict(family="Inter", size=11, color="#64748B"),
                            xaxis=dict(gridcolor="#E2E8F0", zeroline=False),
                            yaxis=dict(gridcolor="#E2E8F0", zeroline=False),
                        )
                        st.markdown('<div class="col-label" style="margin-top:1rem;">Class Distribution</div>', unsafe_allow_html=True)
                        st.plotly_chart(fig, width="stretch")

                    with st.expander("Full JSON Passport"):
                        st.json(passport)

    except ImportError as e:
        st.markdown(
            f"<div class='warn-notice'>Database module unavailable: {e}</div>",
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.exception(e)


with tab_analytics:
    st.markdown(
        '<div class="page-header"><p class="page-title">Fleet Analytics</p>'
        '<p class="page-subtitle">Aggregate statistics and trends across all recorded inspections.</p></div>',
        unsafe_allow_html=True,
    )

    try:
        from src.passport.passport_manager import PassportManager
        import pandas as pd
        import plotly.graph_objects as go
        import plotly.express as px

        pm = PassportManager(db_path=db_path)
        records = pm.list_inspections(limit=500)

        if not records:
            st.markdown(
                """
                <div class="empty-state">
                    <div class="empty-state-title">No data available</div>
                    <div class="empty-state-desc">Run inspections to populate analytics. Data will appear here automatically once records exist.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            df = pd.DataFrame(records)

            total = len(df)
            n_pass = int((df["inspection_status"] == "PASS").sum()) if "inspection_status" in df.columns else 0
            n_review = int((df["inspection_status"] == "REVIEW").sum()) if "inspection_status" in df.columns else 0
            n_flagged = int((df["inspection_status"] == "FLAGGED").sum()) if "inspection_status" in df.columns else 0
            avg_det = df["total_frame_detections"].mean() if "total_frame_detections" in df.columns else 0
            avg_conf = df["avg_confidence"].mean() if "avg_confidence" in df.columns else 0

            st.markdown(
                f"""
                <div class="kpi-strip">
                    <div class="kpi-cell">
                        <div class="kpi-label">Total</div>
                        <div class="kpi-value">{total}</div>
                    </div>
                    <div class="kpi-cell">
                        <div class="kpi-label">Pass</div>
                        <div class="kpi-value pass">{n_pass}</div>
                    </div>
                    <div class="kpi-cell">
                        <div class="kpi-label">Review</div>
                        <div class="kpi-value review">{n_review}</div>
                    </div>
                    <div class="kpi-cell">
                        <div class="kpi-label">Flagged</div>
                        <div class="kpi-value flagged">{n_flagged}</div>
                    </div>
                    <div class="kpi-cell">
                        <div class="kpi-label">Avg Detections</div>
                        <div class="kpi-value">{avg_det:.1f}</div>
                    </div>
                    <div class="kpi-cell">
                        <div class="kpi-label">Avg Confidence</div>
                        <div class="kpi-value">{avg_conf:.3f}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            chart_defaults = dict(
                template="plotly_white",
                plot_bgcolor="#ffffff",
                paper_bgcolor="#ffffff",
                font=dict(family="Inter", size=11, color="#64748B"),
                margin=dict(l=10, r=10, t=28, b=10),
                height=300,
            )

            c1, c2 = st.columns(2)

            with c1:
                if "inspection_status" in df.columns:
                    status_counts = df["inspection_status"].value_counts().reset_index()
                    status_counts.columns = ["Status", "Count"]
                    fig_pie = px.pie(
                        status_counts,
                        names="Status",
                        values="Count",
                        color="Status",
                        color_discrete_map={"PASS": "#22c55e", "REVIEW": "#f59e0b", "FLAGGED": "#ef4444"},
                    )
                    fig_pie.update_traces(
                        textfont=dict(family="Inter", size=11),
                        hole=0.4,
                    )
                    fig_pie.update_layout(
                        **chart_defaults,
                        title=dict(text="Inspection Status", font=dict(size=12, color="#64748B")),
                        legend=dict(font=dict(size=10)),
                    )
                    st.plotly_chart(fig_pie, width="stretch")

            with c2:
                from collections import Counter
                agg_classes: Counter = Counter()
                for row in records:
                    cc_json = row.get("class_counts_json", "{}")
                    try:
                        cc = json.loads(cc_json) if isinstance(cc_json, str) else (cc_json or {})
                        agg_classes.update(cc)
                    except Exception:
                        pass
                if agg_classes:
                    cls_df = pd.DataFrame(agg_classes.most_common(), columns=["Class", "Count"])
                    fig_bar = go.Figure(go.Bar(
                        x=cls_df["Class"],
                        y=cls_df["Count"],
                        marker_color="#06B6D4",
                        marker_line_width=0,
                    ))
                    fig_bar.update_layout(
                        **chart_defaults,
                        title=dict(text="Defect Class Counts", font=dict(size=12, color="#64748B")),
                        xaxis=dict(gridcolor="#E2E8F0", zeroline=False),
                        yaxis=dict(gridcolor="#E2E8F0", zeroline=False),
                    )
                    st.plotly_chart(fig_bar, width="stretch")

            if "start_time" in df.columns:
                df["start_dt"] = pd.to_datetime(df["start_time"], errors="coerce")
                df_sorted = df.sort_values("start_dt").dropna(subset=["start_dt"])
                if not df_sorted.empty:
                    fig_trend = go.Figure(go.Scatter(
                        x=df_sorted["start_dt"],
                        y=df_sorted["total_frame_detections"],
                        mode="lines+markers",
                        line=dict(color="#06B6D4", width=2),
                        marker=dict(color="#06B6D4", size=5),
                    ))
                    fig_trend.update_layout(
                        **chart_defaults,
                        title=dict(text="Detections per Inspection Over Time", font=dict(size=12, color="#64748B")),
                        xaxis=dict(gridcolor="#E2E8F0", zeroline=False),
                        yaxis=dict(gridcolor="#E2E8F0", zeroline=False),
                    )
                    st.plotly_chart(fig_trend, width="stretch")

    except ImportError as e:
        st.markdown(
            f"<div class='warn-notice'>Plotting unavailable: {e}</div>",
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.exception(e)


with tab_system:
    st.markdown(
        '<div class="page-header"><p class="page-title">System Information</p>'
        '<p class="page-subtitle">Hardware configuration and installed package versions for this environment.</p></div>',
        unsafe_allow_html=True,
    )

    try:
        from src.utils.hardware import get_hardware_info
        hw = get_hardware_info()

        gpu_names = hw.get("gpu_names", [])
        cuda_ok = hw.get("cuda_available", False)

        gpu_html = ""
        if gpu_names:
            for g in gpu_names:
                gpu_html += f'<span class="gpu-indicator gpu-on">{g}</span> '
        else:
            gpu_html = '<span class="gpu-indicator gpu-off">No CUDA GPU — CPU only</span>'

        st.markdown(
            f"""
            <div class="hw-grid">
                <div class="hw-section">
                    <div class="hw-section-title">CPU</div>
                    <div class="hw-row"><span class="hw-key">Brand</span><span class="hw-val">{hw.get('cpu_brand', '—')}</span></div>
                    <div class="hw-row"><span class="hw-key">Physical cores</span><span class="hw-val">{hw.get('cpu_cores_physical', '—')}</span></div>
                    <div class="hw-row"><span class="hw-key">Logical cores</span><span class="hw-val">{hw.get('cpu_cores_logical', '—')}</span></div>
                    <div class="hw-row"><span class="hw-key">Clock (MHz)</span><span class="hw-val">{hw.get('cpu_freq_mhz', '—')}</span></div>
                </div>
                <div class="hw-section">
                    <div class="hw-section-title">Memory</div>
                    <div class="hw-row"><span class="hw-key">Total</span><span class="hw-val">{hw.get('ram_total_gb', '—')} GB</span></div>
                    <div class="hw-row"><span class="hw-key">Available</span><span class="hw-val">{hw.get('ram_available_gb', '—')} GB</span></div>
                    <div class="hw-row"><span class="hw-key">Used</span><span class="hw-val">{hw.get('ram_used_pct', 0):.1f}%</span></div>
                </div>
                <div class="hw-section">
                    <div class="hw-section-title">GPU / CUDA</div>
                    <div class="hw-row"><span class="hw-key">Device</span><span class="hw-val">{gpu_html}</span></div>
                    <div class="hw-row"><span class="hw-key">CUDA available</span><span class="hw-val">{'Yes' if cuda_ok else 'No'}</span></div>
                    <div class="hw-row"><span class="hw-key">Device count</span><span class="hw-val">{hw.get('cuda_count', 0)}</span></div>
                </div>
                <div class="hw-section">
                    <div class="hw-section-title">Platform</div>
                    <div class="hw-row"><span class="hw-key">OS</span><span class="hw-val">{hw.get('platform_system', '—')}</span></div>
                    <div class="hw-row"><span class="hw-key">Kernel</span><span class="hw-val">{hw.get('platform_version', '—')}</span></div>
                    <div class="hw-row"><span class="hw-key">Python</span><span class="hw-val">{hw.get('python_version', '—')}</span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    except ImportError as e:
        st.markdown(
            f"<div class='error-notice'>Hardware info unavailable: {e}</div>",
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.exception(e)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown('<div class="col-label">Package Versions</div>', unsafe_allow_html=True)

    try:
        import torch
        import ultralytics
        import cv2 as _cv2
        import numpy as np

        pkgs = {
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "opencv-python": _cv2.__version__,
            "numpy": np.__version__,
        }
        rows = "".join(
            f'<div class="pkg-row"><span class="pkg-name">{k}</span><span class="pkg-ver">{v}</span></div>'
            for k, v in pkgs.items()
        )
        st.markdown(
            f'<div class="hw-section" style="max-width:400px;">{rows}</div>',
            unsafe_allow_html=True,
        )
    except ImportError as e:
        st.markdown(
            f"<div class='warn-notice'>Some packages not installed: {e}</div>",
            unsafe_allow_html=True,
        )
