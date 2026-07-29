import io
import os
import logging
import base64
import threading
from collections import deque, Counter
from datetime import datetime
from typing import Optional, Tuple, List

import av
import cv2
import requests
import numpy as np
import pandas as pd
import streamlit as st
import mediapipe as mp
import skops.io as sio
from gtts import gTTS
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

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
        .result-box {
            border-radius: 20px;
            padding: 1rem;
            border: 1px solid rgba(6,182,212,0.25);
            background: rgba(6,182,212,0.08);
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
    """
    Safely deserialize a trained classifier.

    IMPORTANT: this deliberately does NOT use pickle.loads(). Unpickling
    bytes from a user-uploaded file executes arbitrary code embedded in the
    file (pickle's __reduce__ mechanism) - it is a remote code execution
    vulnerability, not just a data-format risk. skops.io.loads() only
    reconstructs objects from an allow-listed set of safe types, so an
    uploaded model that references anything else is rejected instead of
    silently executed.

    To produce a compatible file, train your classifier as usual and save
    it with:
        import skops.io as sio
        sio.dump(model, "model.skops")
    instead of pickle.dump(...).
    """
    untrusted = sio.get_untrusted_types(data=file_bytes)
    if untrusted:
        raise ValueError(
            "Model file references untrusted object types and was rejected: "
            + ", ".join(untrusted)
        )
    return sio.loads(file_bytes)


def load_model(file) -> Optional[object]:
    if file is None:
        return None
    try:
        return load_model_bytes(file.getvalue())
    except Exception as e:
        st.sidebar.error(f"Model load error: {e}")
        return None


_STUN_ONLY_FALLBACK = [{"urls": ["stun:stun.l.google.com:19302"]}]


def _secret(name: str) -> Optional[str]:
    return st.secrets.get(name, os.environ.get(name))


@st.cache_data(ttl=3600, show_spinner=False)
def get_ice_servers() -> list:
    """
    Returns WebRTC ICE servers for the video call the browser opens to this
    app. A STUN server alone lets two peers discover their public address,
    but many networks - including Streamlit Community Cloud's current setup,
    and most mobile/carrier-grade NAT connections - block the direct
    peer-to-peer connection STUN sets up. A TURN server relays the media
    instead, which works almost everywhere. This checks two ways to get one,
    in order, before giving up and falling back to STUN-only:

    1. Static TURN credentials (secrets: TURN_URLS, TURN_USERNAME,
       TURN_CREDENTIAL). TURN_URLS is comma-separated. This works with:
         - Your own self-hosted coturn server (no third party at all -
           run `coturn` on any VPS with a public IP, open UDP/TCP 3478 and
           a relay port range in the firewall, and set a fixed
           username/password with `lt-cred-mech` in turnserver.conf).
           Guide: https://github.com/coturn/coturn/wiki/turnserver
         - Metered.ca's free plan (metered.ca/tools/openrelay/) if you
           switch its dashboard to "static credentials" mode instead of an
           API key.
       Example secrets.toml:
         TURN_URLS = "turn:your-server.example.com:3478"
         TURN_USERNAME = "someuser"
         TURN_CREDENTIAL = "somepassword"

    2. Metered.ca's free-tier dynamic credentials API (20 GB/month free,
       no credit card required - sign up at metered.ca). Create a TURN app
       in their dashboard, which gives you a subdomain and an API key.
       Example secrets.toml:
         METERED_DOMAIN = "your-subdomain"
         METERED_API_KEY = "..."

    Neither configured -> falls back to STUN-only, which works for local
    testing but is likely to fail intermittently once deployed - a warning
    is shown in the sidebar in that case.
    """
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
    """
    Generate MP3 bytes using gTTS. Cached per (text, lang) so repeated
    phrases (greetings, digits, common signs) don't re-hit the network on
    every rerun - this also means the app needs internet access the first
    time each phrase is spoken.
    """
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
        st.warning(
            f"Speech generation failed ({e}). This usually means no internet "
            "connection is available - gTTS requires one."
        )


def render_audio_player():
    """
    Renders an auto-playing audio tag when audio exists.
    """
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
    """
    Runs on a background thread managed by streamlit-webrtc. Two things to
    keep in mind when touching this class:

    1. Anything read on the main Streamlit thread (label/confidence/features)
       must go through get_last(), which holds the same lock recv() writes
       under. Reading the bare attributes directly - as the original app did
       - is a data race: the main thread can observe a label from one frame
         paired with features from another.
    2. Settings (model, threshold, smoothing window, etc.) are plain
       attributes set from the main thread each rerun; that's fine to leave
       unlocked since they're simple, independently-meaningful values, not a
       tuple that needs to stay consistent with itself.
    """

    def __init__(self):
        self.hands = get_hands()
        self.model = None
        self.conf_thresh = 0.6
        self.show_landmarks = True
        self.process_every_n = 2
        self.smoothing_window = 5

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
        # Resize the deque if the user changed the smoothing window this run.
        if self._recent_labels.maxlen != self.smoothing_window:
            self._recent_labels = deque(self._recent_labels, maxlen=self.smoothing_window)

        self._recent_labels.append(raw_label)
        if len(self._recent_labels) < self._recent_labels.maxlen:
            return raw_label

        most_common, count = Counter(self._recent_labels).most_common(1)[0]
        # Require a real majority before committing to a smoothed label,
        # otherwise fall back to the latest raw reading.
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
                    raw_label, conf = predict_with_model(self.model, features, self.conf_thresh)
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
            help="Higher = less CPU load, lower = more responsive. Raise this on slower laptops.",
        )
        smoothing_window = st.slider(
            "Smoothing window (frames)", 1, 15, 5,
            help="Majority-vote over this many processed frames before a prediction is treated as stable. "
                 "Reduces flicker between similar signs.",
        )

    st.divider()
    st.subheader("🧠 Optional model")
    st.caption(
        "Upload a trained classifier based on 63 hand landmark features, saved with "
        "`skops.io.dump(model, 'model.skops')` (NOT pickle - pickle files are no longer "
        "accepted here because loading one can run arbitrary code)."
    )
    model_file = st.file_uploader("Upload model", type=["skops"])
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
            st.markdown('<span class="demo-badge">⚠ Demo mode - no trained model loaded</span>', unsafe_allow_html=True)
            st.caption(
                "Predictions are coming from a small set of hardcoded heuristic rules, "
                "not a trained classifier. Upload a `.skops` model in the sidebar for real recognition."
            )
        st.caption("Allow webcam access, then show one hand sign clearly.")

        ice_servers = get_ice_servers()
        if ice_servers == _STUN_ONLY_FALLBACK:
            st.sidebar.warning(
                "No TURN server configured - the webcam connection may fail to establish "
                "for visitors outside your local network. See the get_ice_servers() "
                "docstring for free/self-hosted setup options."
            )
            with st.sidebar.expander("TURN diagnostics"):
                st.write(
                    {
                        "TURN_URLS detected": bool(_secret("TURN_URLS")),
                        "TURN_USERNAME detected": bool(_secret("TURN_USERNAME")),
                        "TURN_CREDENTIAL detected": bool(_secret("TURN_CREDENTIAL")),
                        "METERED_DOMAIN detected": bool(_secret("METERED_DOMAIN")),
                        "METERED_API_KEY detected": bool(_secret("METERED_API_KEY")),
                    }
                )
                st.caption(
                    "'detected' means the app found a non-empty value for that secret - it "
                    "doesn't confirm the value is correct. If METERED_DOMAIN/METERED_API_KEY "
                    "both show True here but the connection still fails, the credentials "
                    "themselves are likely wrong or the fetch to Metered failed silently - "
                    "check the app logs (Manage app -> Logs) for a 'Could not fetch Metered "
                    "ICE servers' line."
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

    st.caption("Tip: Good lighting, plain background, and keeping one hand centered will improve results.")

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
        6. Train a classifier and save it with `skops.io.dump(model, "model.skops")`
        7. Upload the `.skops` file back into the app
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

**Pipeline:** Live webcam → MediaPipe hand landmarks → 63 normalized features → majority-vote smoothing → classifier → sign label → optional speech.

**Developed by:** **Ribe Boys Senior School**

**Current limitations**
- Single-hand detection only
- No facial expressions or grammar recognition
- Demo fallback rules are limited compared to a full trained model
- True sign language recognition works better with motion-based sequence models
- Models must be exported with `skops` rather than `pickle` for security reasons
"""
)
