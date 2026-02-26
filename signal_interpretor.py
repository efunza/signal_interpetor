import io
import pickle
import platform
import sys
import traceback
from typing import Optional, Tuple

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="AI Sign Interpreter (Camera)", page_icon="🤟", layout="wide")
st.title("🤟 AI Sign Language Interpreter (Camera)")
st.caption("Cloud-friendly version (no OpenCV in your code). Uses camera snapshots + MediaPipe Hands.")

# -----------------------------
# Sidebar: Debugger + Settings
# -----------------------------
with st.sidebar:
    st.header("🪲 Debugger")
    debug_mode = st.toggle("Enable debug panel", value=True)
    stop_after_debug = st.toggle("Stop app after debug (avoid crash)", value=False)

    st.divider()
    st.header("⚙️ Settings")
    show_landmarks = st.toggle("Draw hand landmarks", value=True)

    st.divider()
    st.subheader("🧠 Optional model (.pkl)")
    st.caption(
        "Upload a classifier (scikit-learn style) that predicts labels from 63 features "
        "(21 landmarks × x,y,z)."
    )
    model_file = st.file_uploader("Upload model", type=["pkl"])
    conf_thresh = st.slider("Min confidence (if predict_proba exists)", 0.0, 1.0, 0.6, 0.05)

# -----------------------------
# Debug utilities
# -----------------------------
def debug_box(title: str, obj):
    with st.expander(title, expanded=True):
        st.write(obj)

def show_environment_debug():
    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "sys_path_head": sys.path[:6],
    }
    debug_box("Environment", info)

def show_import_debug(module_name: str):
    try:
        mod = __import__(module_name)
        data = {
            "module": module_name,
            "__file__": getattr(mod, "__file__", None),
            "__version__": getattr(mod, "__version__", None),
            "dir_has_solutions": ("solutions" in dir(mod)),
            "hasattr_solutions": hasattr(mod, "solutions"),
        }
        debug_box(f"Import debug: {module_name}", data)
        return mod, None
    except Exception as e:
        err = {
            "module": module_name,
            "error": repr(e),
            "traceback": traceback.format_exc(),
        }
        debug_box(f"Import FAILED: {module_name}", err)
        return None, e

# -----------------------------
# Debug panel (runs before mediapipe usage)
# -----------------------------
if debug_mode:
    show_environment_debug()

# Try import mediapipe safely (so you see what's happening)
mp, mp_err = show_import_debug("mediapipe") if debug_mode else (None, None)

if mp is None:
    st.error("MediaPipe could not be imported. Check Streamlit logs and dependencies.")
    st.stop()

# If mediapipe imported but solutions is missing, stop with a helpful message
has_solutions = hasattr(mp, "solutions")
if debug_mode:
    debug_box("MediaPipe API check", {
        "has_mp_solutions": has_solutions,
        "expected_for_this_app": "mp.solutions.hands should exist (legacy Solutions API)",
        "your_next_step_if_false": (
            "Streamlit Cloud is not using mediapipe==0.10.30 OR something is shadowing mediapipe. "
            "Make sure requirements.txt is in the app folder/root, clear cache/rebuild, "
            "and ensure there is no mediapipe.py or mediapipe/ folder in your repo."
        )
    })

if stop_after_debug:
    st.info("Stopped after debug (as requested). Turn off 'Stop app after debug' to continue.")
    st.stop()

if not has_solutions:
    st.error(
        "Your installed mediapipe does NOT have mp.solutions, so this script cannot run.\n\n"
        "Fix: Ensure Streamlit Cloud actually installs mediapipe==0.10.30 (requirements.txt location + rebuild), "
        "and ensure you do not have a mediapipe.py or mediapipe/ in your repo."
    )
    st.stop()

# -----------------------------
# Now safe to use mp.solutions
# -----------------------------
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,     # snapshots
    max_num_hands=1,
    min_detection_confidence=0.6,
)

# -----------------------------
# Model loading
# -----------------------------
def load_model(file) -> Optional[object]:
    if file is None:
        return None
    try:
        return pickle.loads(file.read())
    except Exception as e:
        st.sidebar.error(f"Model load error: {e}")
        return None

MODEL = load_model(model_file)

# -----------------------------
# Helpers
# -----------------------------
def landmarks_to_features(hand_landmarks) -> np.ndarray:
    """
    21 landmarks -> 63 features (x,y,z), normalized:
    - translate so wrist (landmark 0) is origin
    - scale by max 2D distance for size invariance
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
    pts = pts - pts[0]  # wrist origin

    scale = np.max(np.linalg.norm(pts[:, :2], axis=1))
    if scale > 0:
        pts = pts / scale

    return pts.flatten()

def predict_with_model(model, feats: np.ndarray, threshold: float) -> Tuple[str, float]:
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([feats])[0]
            idx = int(np.argmax(proba))
            conf = float(proba[idx])
            label = str(model.classes_[idx]) if hasattr(model, "classes_") else f"CLASS_{idx}"
            if conf < threshold:
                return "UNKNOWN", conf
            return label, conf
        label = model.predict([feats])[0]
        return str(label), 0.5
    except Exception:
        return "UNKNOWN", 0.0

def demo_gesture(hand_landmarks) -> Tuple[str, float]:
    """
    Simple demo gesture rules:
    - OPEN_PALM
    - FIST
    - THUMBS_UP
    NOTE: This is a demo; real sign recognition needs training.
    """
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]

    ext = []
    for tip, pip in zip(tips, pips):
        ext.append(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y)

    thumb, index, middle, ring, pinky = ext

    if index and middle and ring and pinky:
        return "OPEN_PALM", 0.85
    if (not index) and (not middle) and (not ring) and (not pinky):
        return ("FIST", 0.85 if not thumb else 0.75)
    if thumb and (not index) and (not middle) and (not ring) and (not pinky):
        return "THUMBS_UP", 0.85

    return "UNKNOWN", 0.40

def draw_landmarks_pil(img: Image.Image, hand_landmarks) -> Image.Image:
    """Draw landmark points and connections using PIL (no OpenCV)."""
    draw = ImageDraw.Draw(img)
    w, h = img.size

    connections = list(mp_hands.HAND_CONNECTIONS)

    # Draw connections
    for a, b in connections:
        ax = hand_landmarks.landmark[a].x * w
        ay = hand_landmarks.landmark[a].y * h
        bx = hand_landmarks.landmark[b].x * w
        by = hand_landmarks.landmark[b].y * h
        draw.line([(ax, ay), (bx, by)], width=3)

    # Draw points
    for lm in hand_landmarks.landmark:
        x = lm.x * w
        y = lm.y * h
        r = 6
        draw.ellipse([(x - r, y - r), (x + r, y + r)])

    return img

# -----------------------------
# Camera input (snapshot)
# -----------------------------
st.subheader("📷 Take a picture (camera snapshot)")
camera_file = st.camera_input("Show your hand clearly in the frame")

if camera_file is None:
    st.info("Take a snapshot to start interpreting.")
    st.stop()

# Read image
img = Image.open(camera_file).convert("RGB")
img_np = np.ascontiguousarray(np.array(img))

# Run MediaPipe
try:
    results = hands.process(img_np)
except Exception as e:
    st.error(f"MediaPipe processing failed: {e}")
    if debug_mode:
        st.code(traceback.format_exc())
    st.stop()

label, conf = "NO_HAND", 0.0

if results.multi_hand_landmarks:
    hand_lms = results.multi_hand_landmarks[0]
    feats = landmarks_to_features(hand_lms)

    if MODEL is not None:
        label, conf = predict_with_model(MODEL, feats, conf_thresh)
    else:
        label, conf = demo_gesture(hand_lms)

    if show_landmarks:
        img = draw_landmarks_pil(img, hand_lms)
else:
    st.warning("No hand detected. Try brighter light and keep the full hand in frame.")

col1, col2 = st.columns([2, 1])

with col1:
    st.image(img, caption="Processed snapshot", use_container_width=True)

with col2:
    st.metric("Prediction", label)
    st.metric("Confidence", f"{conf:.2f}")
    st.caption("Tip: For better accuracy, keep your full hand in frame with good lighting.")

st.divider()
st.subheader("✅ Science Fair Notes")
st.markdown(
    """
**Pipeline:** Camera → MediaPipe hand landmarks → 63 features → classifier → output label.

**To make it a real interpreter:**  
Train a model on a defined set of signs (e.g., 10 KSL words or ASL letters) and upload the `.pkl`.

**Limitations:**  
Snapshot mode is less smooth than live video. Real sign language also uses two hands + facial expressions + grammar.
"""
)
