import os
import uuid
import time
import tempfile
import threading
import requests
import soundfile as sf
from flask import Flask, request, jsonify, render_template, send_file, session
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# CONFIG
# -----------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "sp-a2-chatbot-local-2026")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_FOLDER = os.path.join(BASE_DIR, "models")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")
INPUT_AUDIO_FOLDER = os.path.join(BASE_DIR, "input_audio")
KOKORO_ONNX = os.path.join(MODELS_FOLDER, "kokoro", "kokoro-v1.0.onnx")
KOKORO_VOICES = os.path.join(MODELS_FOLDER, "kokoro", "voices-v1.0.bin")
KOKORO_ONNX_MIN_SIZE = 50 * 1024 * 1024  # 50 MB

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL   = os.environ.get("OLLAMA_MODEL",    "qwen3:1.7b")
WHISPER_MODEL  = os.environ.get("WHISPER_MODEL",   "turbo")
KOKORO_VOICE   = os.environ.get("KOKORO_VOICE",    "af_sarah")
FLASK_PORT     = int(os.environ.get("PORT",        8080))

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(INPUT_AUDIO_FOLDER, exist_ok=True)
os.makedirs(os.path.join(MODELS_FOLDER, "kokoro"), exist_ok=True)

# Models loaded once, protected by a lock during initialisation
_whisper_model: WhisperModel | None = None
_kokoro_model: Kokoro | None = None
_init_lock = threading.Lock()

# In-memory chat history: session_id -> list[dict]
_chat_history: dict[str, list] = {}

# Kokoro language/voice table (only languages kokoro-onnx v1.0 ships voices for)
_KOKORO_LANG_MAP: dict[str, tuple[str, str]] = {
    "en": ("en-us", KOKORO_VOICE),   # default voice from .env
    "it": ("it",    "if_sara"),
    "fr": ("fr-fr", "ff_siwis"),
}
# Whisper accepts ISO-639-1 codes directly
_WHISPER_LANGS = {"en", "it", "fr", "de"}


# -----------------------------
# MODEL HELPERS
# -----------------------------
def get_whisper_model() -> WhisperModel:
    global _whisper_model
    with _init_lock:
        if _whisper_model is None:
            print("Loading Whisper model…")
            _whisper_model = WhisperModel(
                WHISPER_MODEL,
                device="cpu",
                compute_type="int8",
                download_root=os.path.join(MODELS_FOLDER, "whisper"),
            )
            print("Whisper ready.")
    return _whisper_model


def get_kokoro_model() -> Kokoro:
    global _kokoro_model
    with _init_lock:
        if _kokoro_model is None:
            if not os.path.isfile(KOKORO_ONNX):
                raise RuntimeError(
                    f"Kokoro ONNX not found: {KOKORO_ONNX}\nRun ./run.sh first."
                )
            if os.path.getsize(KOKORO_ONNX) < KOKORO_ONNX_MIN_SIZE:
                os.remove(KOKORO_ONNX)
                raise RuntimeError("Kokoro ONNX corrupted (too small). Re-run ./run.sh.")
            print("Loading Kokoro TTS model…")
            _kokoro_model = Kokoro(KOKORO_ONNX, KOKORO_VOICES)
            print("Kokoro ready.")
    return _kokoro_model


def _ensure_session() -> str:
    """Return (and create if needed) a persistent session ID."""
    if "sid" not in session:
        sid = str(uuid.uuid4())
        session["sid"] = sid
        _chat_history[sid] = []
    sid = session["sid"]
    if sid not in _chat_history:
        _chat_history[sid] = []
    return sid


# -----------------------------
# ROUTES
# -----------------------------
@app.route("/")
def index():
    _ensure_session()
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    sid = _ensure_session()

    if "audio" not in request.files:
        return jsonify({"error": "No audio file in request"}), 400

    audio_file = request.files["audio"]

    # Persist incoming audio to a temp file (ffmpeg-readable by faster-whisper)
    with tempfile.NamedTemporaryFile(
        suffix=".webm", delete=False, dir=INPUT_AUDIO_FOLDER
    ) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    # ── Language params ──────────────────────────────────
    user_lang = request.form.get("user_lang", "en")
    bot_lang  = request.form.get("bot_lang",  "en")

    if user_lang not in _WHISPER_LANGS:
        user_lang = "en"
    if bot_lang not in _KOKORO_LANG_MAP:
        bot_lang = "en"

    kokoro_lang, kokoro_voice = _KOKORO_LANG_MAP[bot_lang]

    bot_lang_names = {"en": "English", "it": "Italian", "fr": "French"}
    bot_lang_name  = bot_lang_names.get(bot_lang, "English")

    try:
        # ── Transcribe ───────────────────────────────────────
        model = get_whisper_model()
        segments, _ = model.transcribe(
            tmp_path,
            language=user_lang,
            beam_size=1,          # greedy – 2-3× faster than default beam_size=5
            vad_filter=True,      # strip leading/trailing silence before decoding
            vad_parameters={"min_silence_duration_ms": 300},
        )
        user_text = " ".join(s.text.strip() for s in segments).strip()

        if not user_text:
            return jsonify({"error": "No speech detected. Please speak clearly and try again."}), 422

        # ── LLM reply ────────────────────────────────────────
        llm_resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            f"You are a smart, friendly, slightly sarcastic voice assistant. "
                            f"Reply in {bot_lang_name} in 1 or 2 short sentences only. "
                            f"Be concise because your answer will be spoken aloud. "
                            f"Do not output reasoning, analysis, or hidden thoughts. "
                            f"Output only the final spoken answer in {bot_lang_name}."
                        ),
                    },
                    {"role": "user", "content": user_text},
                ],
                "stream": False,
                "think": False,
                "options": {"num_predict": 80},  # cap tokens → faster LLM response
            },
            timeout=120,
        )
        llm_resp.raise_for_status()
        bot_text = (llm_resp.json().get("message", {}).get("content", "") or "").strip()

        if not bot_text:
            return jsonify({"error": "LLM returned an empty response."}), 500

        # ── TTS ──────────────────────────────────────────────
        ts = time.strftime("%Y%m%d_%H%M%S")
        audio_filename = f"response_{ts}_{uuid.uuid4().hex[:6]}.wav"
        audio_path = os.path.join(OUTPUT_FOLDER, audio_filename)

        kokoro = get_kokoro_model()
        samples, sample_rate = kokoro.create(bot_text, voice=kokoro_voice, speed=1.0, lang=kokoro_lang)
        sf.write(audio_path, samples, sample_rate)

        # ── Store & return ───────────────────────────────────
        entry = {
            "id": str(uuid.uuid4()),
            "user": user_text,
            "bot": bot_text,
            "audio_url": f"/audio/{audio_filename}",
            "timestamp": time.strftime("%H:%M"),
        }
        _chat_history[sid].append(entry)
        return jsonify(entry)

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.route("/audio/<filename>")
def serve_audio(filename: str):
    # Prevent path traversal attacks
    safe_name = os.path.basename(filename)
    path = os.path.join(OUTPUT_FOLDER, safe_name)
    if not os.path.isfile(path):
        return "Not found", 404
    return send_file(path, mimetype="audio/wav")


@app.route("/history")
def history():
    sid = session.get("sid", "")
    return jsonify(_chat_history.get(sid, []))


@app.route("/new_chat", methods=["POST"])
def new_chat():
    sid = session.get("sid", "")
    if sid and sid in _chat_history:
        _chat_history[sid] = []
    return jsonify({"ok": True})


# -----------------------------
# STARTUP
# -----------------------------
if __name__ == "__main__":
    print("Warming up models (this may take a moment on first run)…")
    try:
        get_whisper_model()
        get_kokoro_model()
    except Exception as exc:
        print(f"[WARN] Model pre-load failed: {exc}")

    print(f"\nServer ready →  http://localhost:{FLASK_PORT}\n")
    app.run(host="127.0.0.1", port=FLASK_PORT, debug=False, threaded=True)
