# app.py
# Streamlit "AI Sign Language Interpreter" (webcam) — demo + ML model plug-in
#
# IMPORTANT:
# - This is a *framework/demo*. Real sign language interpretation needs a trained model + a defined label set.
# - The built-in logic recognizes a few simple *hand gestures* (open palm / fist / thumbs up) as a demo.
# - For real signs (e.g., ASL/KSL letters/words), upload your own trained model (.pkl) in the sidebar.
#
# Run locally:
#   pip install -r requirements.txt
#   streamlit run app.py
#
# Streamlit Cloud:
# - Put requirements.txt in the repo root
# - Some cloud environments may not allow webcam access; test locally if needed.

from __future__ import annotations

import io
import pickle
from dataclasses import dataclass
from typing import Optional, Tuple

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

import mediapipe as mp


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="AI Sign Language Interpreter (Camera)", page_icon="🤟", layout="wide")
st.title("🤟 AI Sign Language Interpreter (Camera)")
st.caption(
    "Webcam-based hand tracking using MediaPipe. Demo recognizes a few simple gestures, "
    "or you can upload a trained model for real sign labels."
)

with st.sidebar:
    st.header("⚙️ Settings")

    show_landmarks = st.toggle("Show hand landmarks", value=True)
    mirror = st.toggle("Mirror camera (selfie)", value=True)

    st.divider()
    st.subheader("🧠 Optional: Upload your trained model (.pkl)")
    st.caption(
        "Upload a scikit-learn style model that supports predict() (and optionally predict_proba()). "
        "The model should accept a 1D feature vector built from 21 hand landmarks (x,y,z) = 63 values."
    )
    model_file = st.file_uploader("Model file (.pkl)", type=["pkl"])

    confidence_threshold = st.slider("Min confidence (if model provides probabilities)", 0.0, 1.0, 0.6, 0.05)

    st.divider()
    st.subheader("ℹ️ Tips")
    st.markdown(
        "- Keep your hand fully visible in the frame.\n"
        "- Good lighting helps.\n"
        "- If using a custom model, train it on the same feature format used here."
    )


# -----------------------------
# Load optional model
# -----------------------------
def load_model(file) -> Optional[object]:
    if file is None:
        return None
    try:
        data = file.read()
        return pickle.loads(data)
    except Exception as e:
        st.sidebar.error(f"Could not load model: {e}")
        return None


MODEL = load_model(model_file)


# -----------------------------
# MediaPipe setup
# -----------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


# -----------------------------
# Feature extraction
# -----------------------------
def landmarks_to_features(hand_landmarks, image_w: int, image_h: int) -> np.ndarray:
    """
    Convert 21 hand landmarks to a normalized feature vector length 63 (x,y,z).
    Normalization:
      - translate so wrist is origin
      - scale by max distance to reduce size variance
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)

    # wrist as origin (landmark 0)
    pts = pts - pts[0]

    # scale
    scale = np.max(np.linalg.norm(pts[:, :2], axis=1))
    if scale > 0:
        pts = pts / scale

    return pts.flatten()  # 63


# -----------------------------
# Demo gesture rules (fallback)
# -----------------------------
def demo_gesture_label(hand_landmarks) -> Tuple[str, float]:
    """
    Very simple heuristic labels for a few *gestures*:
      - OPEN_PALM
      - FIST
      - THUMBS_UP
    Returns (label, confidence-ish score)
    """
    # Landmarks:
    # 0 wrist
    # Thumb tip: 4
    # Index tip: 8
    # Middle tip: 12
    # Ring tip: 16
    # Pinky tip: 20
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]  # just before tips (approx)

    # Basic "finger extended" check: tip y < pip y (hand upright-ish in image coordinates)
    # NOTE: y is downward on image; smaller y means higher.
    ext = []
    for tip, pip in zip(tips, pips):
        ext.append(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y)

    # ext[0]=thumb, ext[1]=index, ext[2]=middle, ext[3]=ring, ext[4]=pinky
    thumb, index, middle, ring, pinky = ext

    # OPEN PALM: most fingers extended
    if index and middle and ring and pinky:
        return "OPEN_PALM", 0.85

    # FIST: most fingers not extended
    if (not index) and (not middle) and (not ring) and (not pinky):
        # if thumb also tucked, stronger fist
        return ("FIST", 0.85 if not thumb else 0.75)

    # THUMBS UP: thumb extended, others tucked
    if thumb and (not index) and (not middle) and (not ring) and (not pinky):
        return "THUMBS_UP", 0.85

    return "UNKNOWN", 0.40


# -----------------------------
# Model prediction wrapper
# -----------------------------
def predict_with_model(model, features: np.ndarray, conf_thresh: float) -> Tuple[str, float]:
    """
    Try predict_proba -> label + confidence.
    Otherwise fallback to predict -> label with confidence=0.5.
    """
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([features])[0]
            idx = int(np.argmax(proba))
            conf = float(proba[idx])
            label = str(model.classes_[idx]) if hasattr(model, "classes_") else f"CLASS_{idx}"
            if conf < conf_thresh:
                return "UNKNOWN", conf
            return label, conf

        label = model.predict([features])[0]
        return str(label), 0.5
    except Exception:
        return "UNKNOWN", 0.0


# -----------------------------
# Video processor (streamlit-webrtc)
# -----------------------------
@dataclass
class AppState:
    last_label: str = "—"
    last_conf: float = 0.0


if "app_state" not in st.session_state:
    st.session_state.app_state = AppState()


class HandSignProcessor:
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        if mirror:
            img = cv2.flip(img, 1)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)

        label = "NO_HAND"
        conf = 0.0

        if res.multi_hand_landmarks:
            hand_lms = res.multi_hand_landmarks[0]

            # Draw landmarks
            if show_landmarks:
                mp_draw.draw_landmarks(
                    img,
                    hand_lms,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

            # Extract features
            h, w = img.shape[:2]
            feats = landmarks_to_features(hand_lms, w, h)

            # Predict
            if MODEL is not None:
                label, conf = predict_with_model(MODEL, feats, confidence_threshold)
            else:
                label, conf = demo_gesture_label(hand_lms)

        # Store
        st.session_state.app_state.last_label = label
        st.session_state.app_state.last_conf = conf

        # Overlay label
        overlay = f"{label}  ({conf:.2f})"
        cv2.rectangle(img, (10, 10), (10 + 320, 60), (0, 0, 0), -1)
        cv2.putText(img, overlay, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# Some school networks block WebRTC; use public STUN server for better connectivity.
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.subheader("📷 Live Camera")
st.write(
    "If your browser asks for camera permission, allow it. "
    "If WebRTC doesn’t work on Streamlit Cloud, run locally or try a different network."
)

webrtc_ctx = webrtc_streamer(
    key="sign-cam",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=HandSignProcessor,
    async_processing=True,
)

# Display current prediction as text (outside the video)
col1, col2 = st.columns(2)
with col1:
    st.metric("Current label", st.session_state.app_state.last_label)
with col2:
    st.metric("Confidence", f"{st.session_state.app_state.last_conf:.2f}")

st.divider()
st.subheader("✅ What to show on your science fair board")
st.markdown(
    "- **Problem:** Communication barriers for deaf/hard-of-hearing users.\n"
    "- **Method:** Webcam → MediaPipe hand landmarks → features → classifier → predicted label.\n"
    "- **Data:** Your labeled sign dataset (images/video frames) used to train the model.\n"
    "- **Evaluation:** Accuracy, confusion matrix, performance in different lighting/backgrounds.\n"
    "- **Limitations:** Needs good camera view; sign languages have grammar and two-hand + facial cues."
)

st.subheader("📦 If you want *real sign language* (not just gestures)")
st.markdown(
    "Train a model on your chosen sign set (e.g., KSL/ASL letters or common school phrases) and upload it.\n\n"
    "**Feature format this app expects:** 21 landmarks × (x,y,z) = **63 numbers**, normalized by wrist + scale."
)
