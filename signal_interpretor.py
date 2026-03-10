import threading
import pickle
from datetime import datetime
from typing import Optional, Tuple, List

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import mediapipe as mp
import streamlit.components.v1 as components
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase


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
if "dataset_rows" not in st.session_state:
    st.session_state.dataset_rows = []

if "history" not in st.session_state:
    st.session_state.history = []

if "last_spoken_label" not in st.session_state:
    st.session_state.last_spoken_label = ""

if "last_auto_added_label" not in st.session_state:
    st.session_state.last_auto_added_label = ""


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
        .result-box {
            border-radius: 20px;
            padding: 1rem;
            border: 1px solid rgba(6,182,212,0.25);
            background: rgba(6,182,212,0.08);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Cached resources
# -----------------------------
@st.cache_resource
def get_hands():
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )


@st.cache_resource
def load_model_bytes(file_bytes: bytes):
    return pickle.loads(file_bytes)


def load_model(file) -> Optional[object]:
    if file is None:
        return None
    try:
        return load_model_bytes(file.getvalue())
    except Exception as e:
        st.sidebar.error(f"Model load error: {e}")
        return None


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


def predict_with_model(model, feats: np.ndarray, threshold: float) -> Tuple[str, float]:
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([feats])[0]
            idx = int(np.argmax(proba))
            conf = float(proba[idx])
            label = str(model.classes_[idx]) if hasattr(model, "classes_") else f"CLASS_{idx}"
            return (label, conf) if conf >= threshold else ("UNKNOWN", conf)

        label = model.predict([feats])[0]
        return str(label), 0.50
    except Exception:
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


def speak_text(text: str):
    if not text:
        return
    safe = text.replace("\\", "\\\\").replace("'", "\\'")
    components.html(
        f"""
        <script>
            const msg = new SpeechSynthesisUtterance('{safe}');
            msg.rate = 0.95;
            msg.pitch = 1.0;
            window.speechSynthesis.cancel();
            window.speechSynthesis.speak(msg);
        </script>
        """,
        height=0,
    )


def maybe_add_to_history(label: str):
    if label in ["NO_HAND", "UNKNOWN"]:
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

        self.last_label = "NO_HAND"
        self.last_conf = 0.0
        self.last_features = None

        self.lock = threading.Lock()
        self.frame_count = 0
        self.process_every_n = 2

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image = cv2.resize(image, (FRAME_WIDTH, FRAME_HEIGHT))

        self.frame_count += 1

        if self.frame_count % self.process_every_n == 0:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            label, conf = "NO_HAND", 0.0
            features = None

            if results.multi_hand_landmarks:
                hand_lms = results.multi_hand_landmarks[0]
                features = landmarks_to_features(hand_lms)

                if self.model is not None:
                    label, conf = predict_with_model(self.model, features, self.conf_thresh)
                else:
                    label, conf = demo_gesture(hand_lms)

                if self.show_landmarks:
                    image = draw_landmarks_cv2(image, hand_lms)

            with self.lock:
                self.last_label = label
                self.last_conf = conf
                self.last_features = features
        else:
            with self.lock:
                label = self.last_label
                conf = self.last_conf

        cv2.putText(
            image,
            f"{self.last_label} ({self.last_conf:.2f})",
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
    st.subheader("🧠 Optional model (.pkl)")
    st.caption("Upload a trained classifier based on 63 hand landmark features.")
    model_file = st.file_uploader("Upload model", type=["pkl"])
    conf_thresh = st.slider("Minimum confidence", 0.0, 1.0, 0.60, 0.05)

    st.divider()
    if st.button("🧹 Clear phrase/history", use_container_width=True):
        st.session_state.history = []
        st.session_state.last_auto_added_label = ""
        st.rerun()

    if collect_mode and len(st.session_state.dataset_rows) >= MAX_DATASET_ROWS:
        st.warning("Dataset memory cap reached. Old samples will be replaced by newer ones.")

MODEL = load_model(model_file)

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
            <span class="pill">Student-built innovation</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["🎥 Live Interpret", "📘 Learn Signs", "🧪 Data Collection"])

# -----------------------------
# Tab 1: Live Interpret
# -----------------------------
with tab1:
    left, right = st.columns([1.6, 1])

    with left:
        st.subheader("Live camera")
        st.caption("Allow webcam access, then show one hand sign clearly.")

        ctx = webrtc_streamer(
            key="ribesign-live",
            video_processor_factory=SignVideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        if ctx.video_processor:
            ctx.video_processor.model = MODEL
            ctx.video_processor.conf_thresh = conf_thresh
            ctx.video_processor.show_landmarks = show_landmarks

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Phrase Builder")
        current_sentence = sentence_text(st.session_state.history)
        st.write(current_sentence if current_sentence else "_No signs added yet_")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("➕ Add current sign", use_container_width=True):
                if ctx.video_processor:
                    label_now = ctx.video_processor.last_label
                    if label_now not in ["NO_HAND", "UNKNOWN"]:
                        maybe_add_to_history(label_now)
                        st.rerun()

        with c2:
            if st.button("🗑 Reset phrase", use_container_width=True):
                st.session_state.history = []
                st.session_state.last_auto_added_label = ""
                st.rerun()

        current_sentence = sentence_text(st.session_state.history)
        if current_sentence and st.button("🔊 Speak phrase", use_container_width=True):
            speak_text(current_sentence)

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Live Results")
    col1, col2, col3, col4 = st.columns(4)

    label_out, conf_out, feats = "NO_HAND", 0.0, None
    if ctx.video_processor:
        label_out = ctx.video_processor.last_label
        conf_out = ctx.video_processor.last_conf
        feats = ctx.video_processor.last_features

    if auto_add_phrase and append_history:
        if label_out not in ["NO_HAND", "UNKNOWN"]:
            if label_out != st.session_state.last_auto_added_label:
                maybe_add_to_history(label_out)
                st.session_state.last_auto_added_label = label_out
        elif label_out in ["NO_HAND", "UNKNOWN"]:
            st.session_state.last_auto_added_label = ""

    with col1:
        st.metric("Detected sign", label_out)
    with col2:
        st.metric("Confidence", f"{conf_out:.2f}")
    with col3:
        st.metric("Phrase length", len(st.session_state.history))
    with col4:
        if st.button("🔊 Speak current sign", use_container_width=True):
            if label_out not in ["NO_HAND", "UNKNOWN"]:
                speak_text(label_out)

    st.caption("Tip: Good lighting, plain background, and keeping one hand centered will improve results.")

    if auto_speak:
        if label_out not in ["NO_HAND", "UNKNOWN"]:
            if label_out != st.session_state.last_spoken_label:
                speak_text(label_out)
                st.session_state.last_spoken_label = label_out
        else:
            st.session_state.last_spoken_label = ""

    if collect_mode:
        st.divider()
        st.subheader("Training Data Controls")

        if feats is None:
            st.info("Start the webcam and show a detectable sign before saving samples.")
        else:
            c1, c2, c3 = st.columns([1.1, 1.1, 2])

            with c1:
                if st.button("➕ Save sample(s)", use_container_width=True):
                    trimmed = add_samples_to_dataset(selected_label, feats, samples_per_click)
                    if trimmed:
                        st.warning(
                            f"Saved {samples_per_click} sample(s) for {selected_label}. "
                            "Oldest samples were removed to stay within the memory limit."
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
    st.caption("Browse supported signs for learning, testing, and dataset creation.")

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
        This app can also be used to collect live sign samples for training a more accurate classifier.

        **Suggested workflow**
        1. Turn on **Enable data collection** in the sidebar  
        2. Start the webcam  
        3. Choose a target label  
        4. Hold the sign clearly and save samples  
        5. Download the CSV  
        6. Train a classifier and upload the `.pkl` model back into the app
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

**Pipeline:** Live webcam → MediaPipe hand landmarks → 63 normalized features → classifier → sign label → optional speech.

**Developed by:** **Ribe Boys Senior School**

**Current limitations**
- Single-hand detection only
- No facial expressions or grammar recognition
- Demo fallback rules are limited compared to a full trained model
- True sign language recognition works better with motion-based sequence models
"""
)
