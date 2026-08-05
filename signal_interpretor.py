import io
import os
import logging
import base64
import threading
from collections import deque, Counter
from datetime import datetime
from typing import Optional, Tuple, List, Any

import av
import cv2
import requests
import numpy as np
import pandas as pd
import streamlit as st
import mediapipe as mp
import skops.io as sio
import joblib
import pickle
from gtts import gTTS
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# Optional backends
try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import torch
except ImportError:
    torch = None

logger = logging.getLogger(__name__)

# -----------------------------
# App config
# -----------------------------
st.set_page_config(
    page_title="RibeSign AI Live",
    page_icon="🤟",
    layout="wide",
)

# -----------------------------
# Constants
# -----------------------------
APP_NAME = "RibeSign AI"
APP_TAGLINE = "Kenya Sign Language Interpreter"
APP_SUBTEXT = "Built by Ribe Boys Senior School"

MAX_DATASET_ROWS = 3000
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
NEUTRAL_LABELS = ("NO_HAND", "UNKNOWN")

TTS_LANGUAGES = {"English": "en", "Kiswahili": "sw"}

KSL_SIGNS = [
    "HELLO", "THANK_YOU", "PLEASE", "HELP", "STOP", "YES", "NO", "LOVE", "OK", "YOU",
    "A", "B", "C", "D", "I", "J", "M", "N", "Y", "Z",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
]

DEFAULT_LABELS = [
    "HELLO", "YES", "NO", "THANK_YOU", "PLEASE", "HELP", "STOP", "LOVE", "OK", "YOU"
]

SIGN_DESCRIPTIONS = {
    "HELLO": "Greeting sign with an open hand.",
    "THANK_YOU": "A sign often made from the mouth or chin outward.",
    "PLEASE": "A polite gesture used in requests.",
    "HELP": "A support-related sign.",
    "STOP": "A hand sign used to signal stopping.",
    "YES": "An affirmative sign.",
    "NO": "A negative sign.",
    "LOVE": "A sign used to express love or affection.",
    "OK": "A sign showing agreement or approval.",
    "YOU": "A pointing sign directed outward.",
    "A": "Alphabet sign for A.",
    "B": "Alphabet sign for B.",
    "C": "Alphabet sign for C.",
    "D": "Alphabet sign for D.",
    "I": "Alphabet sign for I.",
    "J": "Alphabet sign for J.",
    "M": "Alphabet sign for M.",
    "N": "Alphabet sign for N.",
    "Y": "Alphabet sign for Y.",
    "Z": "Alphabet sign for Z.",
    "0": "Number sign for zero.",
    "1": "Number sign for one.",
    "2": "Number sign for two.",
    "3": "Number sign for three.",
    "4": "Number sign for four.",
    "5": "Number sign for five.",
    "6": "Number sign for six.",
    "7": "Number sign for seven.",
    "8": "Number sign for eight.",
    "9": "Number sign for nine.",
}

mp_hands = mp.solutions.hands

# -----------------------------
# Session state
# -----------------------------
DEFAULTS = {
    "dataset_rows": [],
    "history": [],
    "last_spoken_label": "",
    "last_auto_added_label": "",
    "audio_bytes": None,
    "audio_nonce": 0,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }
        .hero {
            padding: 1.5rem;
            border-radius: 24px;
            background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 55%, #06b6d4 100%);
            color: white;
            margin-bottom: 1rem;
        }
        .hero h1 {
            margin: 0;
            font-size: 2.3rem;
        }
        .hero p {
            margin: 0.35rem 0;
            opacity: 0.96;
        }
        .pill {
            display: inline-block;
            padding: 0.3rem 0.7rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.15);
            margin-right: 0.45rem;
            margin-top: 0.45rem;
            font-size: 0.9rem;
        }
        .card {
            border: 1px solid rgba(148,163,184,0.22);
            border-radius: 20px;
            padding: 1rem;
            background: rgba(255,255,255,0.6);
        }
        .demo-badge {
            display: inline-block;
            padding: 0.2rem 0.6rem;
            border-radius: 999px;
            background: rgba(234,179,8,0.18);
            color: #92400e;
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Cached resources & Multi-format loaders
# -----------------------------
@st.cache_resource
def get_hands():
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )


def load_model_from_file(uploaded_file) -> Optional[Tuple[str, Any]]:
    if uploaded_file is None:
        return None

    file_bytes = uploaded_file.getvalue()
    filename = uploaded_file.name.lower()

    try:
        if filename.endswith(".skops"):
            untrusted = sio.get_untrusted_types(data=file_bytes)
            if untrusted:
                st.sidebar.error(
                    "Skops model rejected due to untrusted types: " + ", ".join(untrusted)
                )
                return None
            return ("sklearn", sio.loads(file_bytes))

        elif filename.endswith((".joblib", ".sav")):
            model = joblib.load(io.BytesIO(file_bytes))
            return ("sklearn", model)

        elif filename.endswith((".pkl", ".pickle")):
            model = pickle.loads(file_bytes)
            return ("sklearn", model)

        elif filename.endswith(".onnx"):
            if ort is None:
                st.sidebar.error("ONNX Runtime is not installed (`pip install onnxruntime`).")
                return None
            session = ort.InferenceSession(file_bytes)
            return ("onnx", session)

        elif filename.endswith((".pt", ".pth")):
            if torch is None:
                st.sidebar.error("PyTorch is not installed (`pip install torch`).")
                return None
            buffer = io.BytesIO(file_bytes)
            try:
                model = torch.jit.load(buffer, map_location=torch.device("cpu"))
            except Exception:
                buffer.seek(0)
                model = torch.load(buffer, map_location=torch.device("cpu"))
            if hasattr(model, "eval"):
                model.eval()
            return ("torch", model)

        else:
            st.sidebar.error("Unsupported file extension.")
            return None

    except Exception as e:
        st.sidebar.error(f"Failed to load model ({uploaded_file.name}): {e}")
        return None


_STUN_ONLY_FALLBACK = [{"urls": ["stun:stun.l.google.com:19302"]}]


def _secret(name: str) -> Optional[str]:
    # Safe access – works even when secrets.toml does not exist
    try:
        return st.secrets.get(name, os.environ.get(name))
    except Exception:
        return os.environ.get(name)


@st.cache_data(ttl=3600, show_spinner=False)
def get_ice_servers() -> list:
    turn_urls = _secret("TURN_URLS")
    turn_username = _secret("TURN_USERNAME")
    turn_credential = _secret("TURN_CREDENTIAL")

    if turn_urls and turn_username and turn_credential:
        urls = [u.strip() for u in turn_urls.split(",") if u.strip()]
        return [{"urls": urls, "username": turn_username, "credential": turn_credential}]

    metered_domain = _secret("METERED_DOMAIN")
    metered_api_key = _secret("METERED_API_KEY")

    if metered_domain and metered_api_key:
        try:
            resp = requests.get(
                f"https://{metered_domain}.metered.live/api/v1/turn/credentials",
                params={"apiKey": metered_api_key},
                timeout=5,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.warning("Could not fetch Metered ICE servers, falling back to STUN-only: %s", e)

    return _STUN_ONLY_FALLBACK


# -----------------------------
# Speech helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def tts_bytes(text: str, lang: str = "en") -> bytes:
    fp = io.BytesIO()
    tts = gTTS(text=text, lang=lang)
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp.read()


def queue_speech(text: str, lang: str = "en"):
    if not text:
        return
    try:
        st.session_state.audio_bytes = tts_bytes(text, lang)
        st.session_state.audio_nonce += 1
    except Exception as e:
        st.warning(f"Speech generation failed ({e}). Check internet connection for gTTS.")


def render_audio_player():
    audio_bytes = st.session_state.get("audio_bytes")
    if not audio_bytes:
        return
    b64 = base64.b64encode(audio_bytes).decode("utf-8")
    nonce = st.session_state.get("audio_nonce", 0)
    st.markdown(
        f"""
        <audio id="tts-player-{nonce}" autoplay controls style="width:100%;">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mpeg">
        </audio>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# Helpers
# -----------------------------
def landmarks_to_features(hand_landmarks) -> np.ndarray:
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
    pts -= pts[0]
    scale = float(np.max(np.linalg.norm(pts[:, :2], axis=1)))
    if scale > 0:
        pts /= scale
    return pts.flatten()


def predict_with_model(
    model_wrapper: Tuple[str, Any],
    feats: np.ndarray,
    threshold: float,
    target_labels: List[str],
) -> Tuple[str, float]:
    if model_wrapper is None:
        return "UNKNOWN", 0.0

    model_type, model = model_wrapper

    try:
        if model_type == "sklearn":
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba([feats])[0]
                idx = int(np.argmax(proba))
                conf = float(proba[idx])
                label = (
                    str(model.classes_[idx])
                    if hasattr(model, "classes_")
                    else (target_labels[idx] if idx < len(target_labels) else f"CLASS_{idx}")
                )
                return (label, conf) if conf >= threshold else ("UNKNOWN", conf)
            label = model.predict([feats])[0]
            return str(label), 0.50

        elif model_type == "onnx":
            input_name = model.get_inputs()[0].name
            input_data = np.expand_dims(feats, axis=0).astype(np.float32)
            outputs = model.run(None, {input_name: input_data})

            raw_out = outputs[0]
            if isinstance(raw_out, list) and isinstance(raw_out[0], dict):
                probs_dict = raw_out[0]
                best_class = max(probs_dict, key=probs_dict.get)
                conf = float(probs_dict[best_class])
                return (str(best_class), conf) if conf >= threshold else ("UNKNOWN", conf)

            probs = np.array(raw_out)[0]
            if probs.ndim > 0 and len(probs) > 1:
                if np.max(probs) > 1.0 or np.min(probs) < 0.0:
                    exp_p = np.exp(probs - np.max(probs))
                    probs = exp_p / exp_p.sum()
                idx = int(np.argmax(probs))
                conf = float(probs[idx])
                label = target_labels[idx] if idx < len(target_labels) else str(idx)
                return (label, conf) if conf >= threshold else ("UNKNOWN", conf)
            return str(raw_out[0]), 0.50

        elif model_type == "torch":
            with torch.no_grad():
                tensor_in = torch.tensor([feats], dtype=torch.float32)
                out = model(tensor_in)
                probs = torch.softmax(out, dim=1)[0]
                conf, idx = torch.max(probs, dim=0)
                conf_val = float(conf.item())
                idx_val = int(idx.item())
                label = target_labels[idx_val] if idx_val < len(target_labels) else str(idx_val)
                return (label, conf_val) if conf_val >= threshold else ("UNKNOWN", conf_val)

    except Exception as e:
        logger.error(f"Inference error ({model_type}): {e}")
        return "UNKNOWN", 0.0

    return "UNKNOWN", 0.0


def demo_gesture(hand_landmarks) -> Tuple[str, float]:
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]
    extended = []
    for tip, pip in zip(tips, pips):
        extended.append(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y)

    thumb, index, middle, ring, pinky = extended

    if index and middle and ring and pinky:
        return "HELLO", 0.85
    if (not index) and (not middle) and (not ring) and (not pinky):
        return ("STOP", 0.75 if thumb else 0.85)
    if thumb and (not index) and (not middle) and (not ring) and (not pinky):
        return "YES", 0.78
    if index and (not middle) and (not ring) and (not pinky):
        return "1", 0.80
    if index and middle and (not ring) and (not pinky):
        return "2", 0.80
    if thumb and index and middle and (not ring) and (not pinky):
        return "3", 0.78
    if (not thumb) and index and middle and ring and pinky:
        return "4", 0.78
    if thumb and index and middle and ring and pinky:
        return "5", 0.82
    return "UNKNOWN", 0.40


def draw_landmarks_cv2(image_bgr, hand_landmarks):
    h, w, _ = image_bgr.shape
    for a, b in list(mp_hands.HAND_CONNECTIONS):
        ax = int(hand_landmarks.landmark[a].x * w)
        ay = int(hand_landmarks.landmark[a].y * h)
        bx = int(hand_landmarks.landmark[b].x * w)
        by = int(hand_landmarks.landmark[b].y * h)
        cv2.line(image_bgr, (ax, ay), (bx, by), (0, 255, 255), 2)
    for lm in hand_landmarks.landmark:
        x = int(lm.x * w)
        y = int(lm.y * h)
        cv2.circle(image_bgr, (x, y), 4, (255, 0, 0), -1)
    return image_bgr


def add_samples_to_dataset(label: str, feats: np.ndarray, n: int):
    base = {f"f{i}": float(feats[i]) for i in range(len(feats))}
    for _ in range(n):
        st.session_state.dataset_rows.append({"label": label, **base})
    if len(st.session_state.dataset_rows) > MAX_DATASET_ROWS:
        st.session_state.dataset_rows = st.session_state.dataset_rows[-MAX_DATASET_ROWS:]
        return True
    return False


def maybe_add_to_history(label: str, skip_unknown: bool = True):
    if label == "NO_HAND":
        return
    if skip_unknown and label == "UNKNOWN":
        return
    if not st.session_state.history or st.session_state.history[-1] != label:
        st.session_state.history.append(label)


def sentence_text(items: List[str]) -> str:
    return " ".join(items).strip()


# -----------------------------
# Video processor
# -----------------------------
class SignVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = get_hands()
        self.model = None
        self.conf_thresh = 0.6
        self.show_landmarks = True
        self.process_every_n = 2
        self.smoothing_window = 5
        self.labels_list = DEFAULT_LABELS
        self.last_label = "NO_HAND"
        self.last_conf = 0.0
        self.last_features = None
        self._recent_labels = deque(maxlen=self.smoothing_window)
        self.lock = threading.Lock()
        self.frame_count = 0

    def get_last(self):
        with self.lock:
            return self.last_label, self.last_conf, self.last_features

    def _smoothed_label(self, raw_label: str) -> str:
        if self._recent_labels.maxlen != self.smoothing_window:
            self._recent_labels = deque(self._recent_labels, maxlen=self.smoothing_window)
        self._recent_labels.append(raw_label)
        if len(self._recent_labels) < self._recent_labels.maxlen:
            return raw_label
        most_common, count = Counter(self._recent_labels).most_common(1)[0]
        if count >= (self._recent_labels.maxlen // 2) + 1:
            return most_common
        return raw_label

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image = cv2.resize(image, (FRAME_WIDTH, FRAME_HEIGHT))
        self.frame_count += 1

        if self.frame_count % max(self.process_every_n, 1) == 0:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            raw_label, conf = "NO_HAND", 0.0
            features = None

            if results.multi_hand_landmarks:
                hand_lms = results.multi_hand_landmarks[0]
                features = landmarks_to_features(hand_lms)

                if self.model is not None:
                    raw_label, conf = predict_with_model(
                        self.model, features, self.conf_thresh, self.labels_list
                    )
                else:
                    raw_label, conf = demo_gesture(hand_lms)

                if self.show_landmarks:
                    image = draw_landmarks_cv2(image, hand_lms)

            label = self._smoothed_label(raw_label)

            with self.lock:
                self.last_label = label
                self.last_conf = conf
                self.last_features = features

        display_label, display_conf, _ = self.get_last()
        cv2.putText(
            image,
            f"{display_label} ({display_conf:.2f})",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return av.VideoFrame.from_ndarray(image, format="bgr24")


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    show_landmarks = st.toggle("Draw hand landmarks", value=True)
    auto_speak = st.toggle("Speak prediction aloud", value=False)
    auto_add_phrase = st.toggle("Auto add stable signs to phrase", value=False)
    append_history = st.toggle("Build phrase from predictions", value=True)
    clear_unknowns = st.toggle("Ignore UNKNOWN in phrase", value=True)

    speech_lang_name = st.selectbox("Speech language", list(TTS_LANGUAGES.keys()))
    speech_lang = TTS_LANGUAGES[speech_lang_name]

    with st.expander("Performance & smoothing"):
        process_every_n = st.slider(
            "Process every Nth frame", 1, 5, 2,
            help="Higher = less CPU load, lower = more responsive.",
        )
        smoothing_window = st.slider(
            "Smoothing window (frames)", 1, 15, 5,
            help="Majority-vote over frames to reduce label flickering.",
        )

    st.divider()
    st.subheader("🧠 Optional model")
    st.caption("Upload a model trained on 63 landmark features (.skops, .joblib, .pkl, .onnx, .pt)")

    model_file = st.file_uploader(
        "Upload model",
        type=["skops", "joblib", "pkl", "pickle", "sav", "onnx", "pt", "pth"]
    )
    conf_thresh = st.slider("Minimum confidence", 0.0, 1.0, 0.60, 0.05)

    st.divider()
    st.subheader("📚 Training data")
    collect_mode = st.toggle("Enable data collection", value=False)

    labels = st.multiselect("Labels", KSL_SIGNS, default=DEFAULT_LABELS)
    if not labels:
        labels = DEFAULT_LABELS
        st.warning("At least one label is required.")

    selected_label = st.selectbox("Label to record", labels, index=0)
    samples_per_click = st.slider("Samples saved per click", 1, 5, 2)

    st.divider()
    if st.button("🧹 Clear phrase/history", use_container_width=True):
        st.session_state.history = []
        st.session_state.last_auto_added_label = ""
        st.rerun()

    if collect_mode and len(st.session_state.dataset_rows) >= MAX_DATASET_ROWS:
        st.warning("Dataset memory cap reached. Old samples will be replaced by newer ones.")

MODEL = load_model_from_file(model_file)

# -----------------------------
# Header
# -----------------------------
st.markdown(
    f"""
    <div class="hero">
        <h1>🤟 {APP_NAME}</h1>
        <p>{APP_TAGLINE}</p>
        <p><strong>{APP_SUBTEXT}</strong></p>
        <div>
            <span class="pill">Live webcam interpretation</span>
            <span class="pill">Text output</span>
            <span class="pill">Speech output</span>
            <span class="pill">Multi-Model Support</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

render_audio_player()

tab1, tab2, tab3 = st.tabs(["🎥 Live Interpret", "📘 Learn Signs", "🧪 Data Collection"])

# -----------------------------
# Tab 1: Live Interpret
# -----------------------------
with tab1:
    left, right = st.columns([1.6, 1])

    with left:
        st.subheader("Live camera")

        if MODEL is None:
            st.markdown(
                '<span class="demo-badge">⚠ Demo mode - no trained model loaded</span>',
                unsafe_allow_html=True,
            )
            st.caption(
                "Predictions are coming from hardcoded rules. Upload a supported classifier "
                "in the sidebar for real sign recognition."
            )
        else:
            m_type = MODEL[0].upper()
            st.success(f"Active model format: **{m_type}**")

        ice_servers = get_ice_servers()
        if ice_servers == _STUN_ONLY_FALLBACK:
            st.sidebar.warning(
                "No TURN server configured – connection may fail outside local network."
            )

        ctx = webrtc_streamer(
            key="ribesign-live",
            video_processor_factory=SignVideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration={"iceServers": ice_servers},
            async_processing=True,
        )

        if ctx.video_processor:
            ctx.video_processor.model = MODEL
            ctx.video_processor.conf_thresh = conf_thresh
            ctx.video_processor.show_landmarks = show_landmarks
            ctx.video_processor.process_every_n = process_every_n
            ctx.video_processor.smoothing_window = smoothing_window
            ctx.video_processor.labels_list = labels

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Phrase Builder")

        current_sentence = sentence_text(st.session_state.history)
        st.write(current_sentence if current_sentence else "_No signs added yet_")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("➕ Add current sign", use_container_width=True):
                if ctx.video_processor:
                    label_now, _, _ = ctx.video_processor.get_last()
                    maybe_add_to_history(label_now, skip_unknown=clear_unknowns)
                    st.rerun()
        with c2:
            if st.button("🗑 Reset phrase", use_container_width=True):
                st.session_state.history = []
                st.session_state.last_auto_added_label = ""
                st.rerun()

        current_sentence = sentence_text(st.session_state.history)
        if current_sentence and st.button("🔊 Speak phrase", use_container_width=True):
            queue_speech(current_sentence, speech_lang)
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Live Results")
    col1, col2, col3, col4 = st.columns(4)

    label_out, conf_out, feats = "NO_HAND", 0.0, None
    if ctx.video_processor:
        label_out, conf_out, feats = ctx.video_processor.get_last()

    if auto_add_phrase and append_history:
        if label_out not in NEUTRAL_LABELS:
            if label_out != st.session_state.last_auto_added_label:
                maybe_add_to_history(label_out, skip_unknown=clear_unknowns)
                st.session_state.last_auto_added_label = label_out
        else:
            st.session_state.last_auto_added_label = ""

    with col1:
        st.metric("Detected sign", label_out)
    with col2:
        st.metric("Confidence", f"{conf_out:.2f}")
    with col3:
        st.metric("Phrase length", len(st.session_state.history))
    with col4:
        if st.button("🔊 Speak current sign", use_container_width=True):
            if label_out not in NEUTRAL_LABELS:
                queue_speech(label_out, speech_lang)
                st.rerun()

    if auto_speak:
        if label_out not in NEUTRAL_LABELS:
            if label_out != st.session_state.last_spoken_label:
                queue_speech(label_out, speech_lang)
                st.session_state.last_spoken_label = label_out
                st.rerun()
        else:
            st.session_state.last_spoken_label = ""

    if collect_mode:
        st.divider()
        st.subheader("Training Data Controls")

        if feats is None:
            st.info("Start webcam and display a hand sign to enable saving samples.")
        else:
            c1, c2, c3 = st.columns([1.1, 1.1, 2])
            with c1:
                if st.button("➕ Save sample(s)", use_container_width=True):
                    trimmed = add_samples_to_dataset(selected_label, feats, samples_per_click)
                    if trimmed:
                        st.warning(
                            f"Saved {samples_per_click} sample(s) for {selected_label}. "
                            "Oldest samples removed to maintain capacity limits."
                        )
                    else:
                        st.success(f"Saved {samples_per_click} sample(s) for {selected_label}.")
            with c2:
                if st.button("🗑 Clear dataset", use_container_width=True):
                    st.session_state.dataset_rows = []
                    st.warning("Dataset cleared.")
            with c3:
                total = len(st.session_state.dataset_rows)
                st.write(f"Total samples saved: **{total} / {MAX_DATASET_ROWS}**")
                if total > 0:
                    df = pd.DataFrame(st.session_state.dataset_rows)
                    counts = df["label"].value_counts().reindex(labels, fill_value=0)
                    st.dataframe(counts.rename("count"), use_container_width=True)
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Download dataset CSV",
                        data=csv,
                        file_name=f"ribesign_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

# -----------------------------
# Tab 2: Learn Signs
# -----------------------------
with tab2:
    st.subheader("Sign Library")
    search = st.text_input("Search sign", placeholder="Try: hello, yes, 2, A...")
    search_value = search.strip().lower()

    if search_value:
        filtered = [
            s for s in KSL_SIGNS
            if search_value in s.lower() or search_value in SIGN_DESCRIPTIONS.get(s, "").lower()
        ]
    else:
        filtered = KSL_SIGNS

    cols = st.columns(3)
    for i, sign in enumerate(filtered):
        with cols[i % 3]:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"### {sign}")
            st.write(SIGN_DESCRIPTIONS.get(sign, "Supported sign target"))
            st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Tab 3: Data Collection
# -----------------------------
with tab3:
    st.subheader("Build and improve the model")
    st.markdown(
        """
        **Suggested workflow**
        1. Turn on **Enable data collection** in the sidebar
        2. Start the webcam
        3. Save landmark samples for signs
        4. Download the dataset CSV
        5. Train a classifier in **Skops, Scikit-learn, ONNX, or PyTorch**
        6. Upload your trained `.skops`, `.pkl`, `.joblib`, `.onnx`, or `.pt` model back into the app!
        """
    )

    total = len(st.session_state.dataset_rows)
    st.metric("Collected samples in session", total)

    if total > 0:
        df_preview = pd.DataFrame(st.session_state.dataset_rows).head(50)
        st.dataframe(df_preview, use_container_width=True)

# -----------------------------
# Footer
# -----------------------------
st.divider()
st.subheader("About this project")
st.markdown(
    """
**RibeSign AI** is a school technology project for sign interpretation using computer vision.

**Pipeline:** Webcam → MediaPipe hand landmarks → 63 normalized features → Multi-format model inference → Majority-vote smoothing → Sign label & Speech synthesis.

**Developed by:** **Ribe Boys Senior School**
"""
)
