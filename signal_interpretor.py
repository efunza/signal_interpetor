import pickle
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

import mediapipe as mp


# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="AI Sign Interpreter (Camera)", page_icon="🤟", layout="wide")
st.title("🤟 AI Sign Language Interpreter (Camera)")
st.caption("Camera snapshot → MediaPipe hand landmarks → features → (optional) ML model → label")


# -----------------------------
# Sidebar
# -----------------------------
DEFAULT_LABELS = ["HELLO", "YES", "NO", "THANK_YOU", "PLEASE", "HELP", "STOP", "LOVE", "OK", "YOU"]

with st.sidebar:
    st.header("⚙️ Settings")
    show_landmarks = st.toggle("Draw hand landmarks", value=True)

    st.divider()
    st.subheader("📚 Data collection (for training)")
    collect_mode = st.toggle("Enable data collection", value=False)
    labels = st.multiselect("Labels (edit if you want)", DEFAULT_LABELS, default=DEFAULT_LABELS)
    if not labels:
        st.warning("Select at least 1 label for data collection.")
        labels = DEFAULT_LABELS

    selected_label = st.selectbox("Label to record", labels, index=0)
    samples_per_click = st.slider("Samples saved per click", 1, 10, 3)

    st.divider()
    st.subheader("🧠 Optional model (.pkl)")
    st.caption(
        "Upload a classifier that predicts labels from 63 features (21 landmarks × x,y,z). "
        "If you don't upload a model, the app uses simple demo rules."
    )
    model_file = st.file_uploader("Upload model", type=["pkl"])
    conf_thresh = st.slider("Min confidence (if predict_proba exists)", 0.0, 1.0, 0.6, 0.05)


# -----------------------------
# Session state (dataset)
# -----------------------------
if "dataset_rows" not in st.session_state:
    st.session_state.dataset_rows = []  # list[dict]


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
# MediaPipe Hands (snapshot mode)
# -----------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,   # snapshots
    max_num_hands=1,
    min_detection_confidence=0.6,
)


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
    scale = float(np.max(np.linalg.norm(pts[:, :2], axis=1)))
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
            return (label, conf) if conf >= threshold else ("UNKNOWN", conf)
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
    """
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]

    extended = []
    for tip, pip in zip(tips, pips):
        extended.append(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y)

    thumb, index, middle, ring, pinky = extended

    if index and middle and ring and pinky:
        return "OPEN_PALM", 0.85
    if (not index) and (not middle) and (not ring) and (not pinky):
        return ("FIST", 0.85 if not thumb else 0.75)
    if thumb and (not index) and (not middle) and (not ring) and (not pinky):
        return "THUMBS_UP", 0.85

    return "UNKNOWN", 0.40


def draw_landmarks_pil(img: Image.Image, hand_landmarks) -> Image.Image:
    """Draw landmark points and connections using PIL."""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    connections = list(mp_hands.HAND_CONNECTIONS)

    # connections
    for a, b in connections:
        ax = hand_landmarks.landmark[a].x * w
        ay = hand_landmarks.landmark[a].y * h
        bx = hand_landmarks.landmark[b].x * w
        by = hand_landmarks.landmark[b].y * h
        draw.line([(ax, ay), (bx, by)], width=3)

    # points
    for lm in hand_landmarks.landmark:
        x = lm.x * w
        y = lm.y * h
        r = 6
        draw.ellipse([(x - r, y - r), (x + r, y + r)])

    return img


def add_samples_to_dataset(label: str, feats: np.ndarray, n: int):
    # store features as f0..f62 for easy training
    base = {f"f{i}": float(feats[i]) for i in range(len(feats))}
    for _ in range(n):
        st.session_state.dataset_rows.append({"label": label, **base})


# -----------------------------
# UI: Camera input (snapshot)
# -----------------------------
st.subheader("📷 Take a picture (camera snapshot)")
camera_file = st.camera_input("Show your hand clearly in the frame")

if camera_file is None:
    st.info("Take a snapshot to start interpreting.")
    st.stop()

img = Image.open(camera_file).convert("RGB")
img_np = np.ascontiguousarray(np.array(img))  # safer in some environments

# Run MediaPipe
results = hands.process(img_np)

label_out, conf_out = "NO_HAND", 0.0
hand_lms = None
feats = None

if results.multi_hand_landmarks:
    hand_lms = results.multi_hand_landmarks[0]
    feats = landmarks_to_features(hand_lms)

    # Prediction
    if MODEL is not None:
        label_out, conf_out = predict_with_model(MODEL, feats, conf_thresh)
    else:
        label_out, conf_out = demo_gesture(hand_lms)

    # Optional drawing
    if show_landmarks:
        img = draw_landmarks_pil(img, hand_lms)
else:
    st.warning("No hand detected. Try brighter light and keep your full hand in frame.")


# -----------------------------
# Data collection UI (main page)
# -----------------------------
if collect_mode:
    st.subheader("📚 Data collection controls")

    if feats is None:
        st.info("Take a snapshot where a hand is detected, then you can save samples.")
    else:
        c1, c2, c3 = st.columns([1.2, 1.2, 2])

        with c1:
            if st.button("➕ Save sample(s)", use_container_width=True):
                add_samples_to_dataset(selected_label, feats, samples_per_click)
                st.success(f"Saved {samples_per_click} sample(s) for {selected_label}.")

        with c2:
            if st.button("🗑️ Clear dataset", use_container_width=True):
                st.session_state.dataset_rows = []
                st.warning("Dataset cleared.")

        with c3:
            total = len(st.session_state.dataset_rows)
            st.write(f"Total samples saved: **{total}**")

            if total > 0:
                df = pd.DataFrame(st.session_state.dataset_rows)

                # show counts per label
                counts = df["label"].value_counts().reindex(labels, fill_value=0)
                st.write("Samples per label:")
                st.dataframe(counts.rename("count"), use_container_width=True)

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download dataset CSV",
                    data=csv,
                    file_name=f"sign_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )


# -----------------------------
# Output display
# -----------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.image(img, caption="Processed snapshot", use_container_width=True)

with col2:
    st.metric("Prediction", label_out)
    st.metric("Confidence", f"{conf_out:.2f}")
    st.caption("Tip: Good lighting + full hand in frame improves detection.")


# -----------------------------
# Notes
# -----------------------------
st.divider()
st.subheader("✅ Science Fair Notes")
st.markdown(
    """
**Pipeline:** Camera → MediaPipe hand landmarks → 63 features → classifier → output label.

**How to build a real demo model:**
1. Turn on **Data collection**
2. Save ~100–200 samples per word (different angles/lighting)
3. Download CSV
4. Train a scikit-learn model and upload the `.pkl`

**Limitations:**  
Snapshot mode is less smooth than live video. Real sign language also uses two hands + facial expressions + grammar.
"""
)
