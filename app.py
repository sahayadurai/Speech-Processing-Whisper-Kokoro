import os
import uuid
import time
import tempfile
import threading
import requests
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# CONFIG
# -----------------------------
app = FastAPI(title="Voice Chatbot")
SECRET_KEY = os.environ.get("SECRET_KEY", "sp-a2-chatbot-local-2026")

# Middleware
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_FOLDER = os.path.join(BASE_DIR, "models")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")
INPUT_AUDIO_FOLDER = os.path.join(BASE_DIR, "input_audio")
TEMPLATES_FOLDER = os.path.join(BASE_DIR, "templates")
KOKORO_ONNX = os.path.join(MODELS_FOLDER, "kokoro", "kokoro-v1.0.onnx")
KOKORO_VOICES = os.path.join(MODELS_FOLDER, "kokoro", "voices-v1.0.bin")
KOKORO_ONNX_MIN_SIZE = 50 * 1024 * 1024  # 50 MB

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL   = os.environ.get("OLLAMA_MODEL",    "qwen3:1.7b")
WHISPER_MODEL  = os.environ.get("WHISPER_MODEL",   "turbo")
KOKORO_VOICE   = os.environ.get("KOKORO_VOICE",    "af_sarah")
FASTAPI_PORT   = int(os.environ.get("PORT",        8080))

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(INPUT_AUDIO_FOLDER, exist_ok=True)
os.makedirs(os.path.join(MODELS_FOLDER, "kokoro"), exist_ok=True)
os.makedirs(TEMPLATES_FOLDER, exist_ok=True)

# Mount static files for templates
try:
    app.mount("/static", StaticFiles(directory=TEMPLATES_FOLDER), name="static")
except Exception:
    pass

# Models loaded once, protected by a lock during initialisation
_whisper_model: WhisperModel | None = None
_kokoro_model: Kokoro | None = None
_init_lock = threading.Lock()

# In-memory chat history: session_id -> list[dict]
_chat_history: dict[str, list] = {}

# Kokoro language/voice table (only languages kokoro-onnx v1.0 ships voices for)
_KOKORO_LANG_MAP: dict[str, tuple[str, str]] = {
    "en": ("en-us", "af_sarah"),    # English - female voice
    "it": ("it",    "if_sara"),     # Italian - female voice
    "fr": ("fr-fr", "ff_siwis"),    # French - female voice
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


def _get_or_create_session_id(request: Request) -> str:
    """Get session ID from cookie or create a new one."""
    if "sid" not in request.session:
        sid = str(uuid.uuid4())
        request.session["sid"] = sid
        _chat_history[sid] = []
    else:
        sid = request.session["sid"]
        if sid not in _chat_history:
            _chat_history[sid] = []
    return sid


# -----------------------------
# ROUTES
# -----------------------------
@app.get("/")
async def index(request: Request):
    _get_or_create_session_id(request)
    return FileResponse(os.path.join(TEMPLATES_FOLDER, "index.html"), media_type="text/html")


@app.get("/test-lang")
async def test_lang(user_lang: str = "en", bot_lang: str = "en"):
    """Test endpoint to verify language parameters"""
    return {
        "received_user_lang": user_lang,
        "received_bot_lang": bot_lang,
        "whisper_langs": list(_WHISPER_LANGS),
        "kokoro_langs": list(_KOKORO_LANG_MAP.keys())
    }


@app.post("/chat")
async def chat(
    request: Request,
    audio: UploadFile = File(...),
    user_lang: str = Form("en"),
    bot_lang: str = Form("en"),
):
    sid = _get_or_create_session_id(request)

    # Persist incoming audio to a temp file (ffmpeg-readable by faster-whisper)
    with tempfile.NamedTemporaryFile(
        suffix=".webm", delete=False, dir=INPUT_AUDIO_FOLDER
    ) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    # ── Language params ──────────────────────────────────
    print(f"\n[DEBUG] === NEW REQUEST ===")
    print(f"[DEBUG] Raw form data received: user_lang={repr(user_lang)}, bot_lang={repr(bot_lang)}")
    
    # Validation with detailed logging
    original_user_lang = user_lang
    original_bot_lang = bot_lang
    
    if user_lang not in _WHISPER_LANGS:
        print(f"[DEBUG] user_lang '{user_lang}' not in {_WHISPER_LANGS}, defaulting to 'en'")
        user_lang = "en"
    if bot_lang not in _KOKORO_LANG_MAP:
        print(f"[DEBUG] bot_lang '{bot_lang}' not in {list(_KOKORO_LANG_MAP.keys())}, defaulting to 'en'")
        bot_lang = "en"

    # Get Kokoro configuration
    kokoro_lang, kokoro_voice = _KOKORO_LANG_MAP[bot_lang]

    bot_lang_names = {"en": "English", "it": "Italian", "fr": "French"}
    bot_lang_name  = bot_lang_names.get(bot_lang, "English")
    
    print(f"[DEBUG] Validation complete:")
    print(f"[DEBUG]   user_lang: {original_user_lang} → {user_lang}")
    print(f"[DEBUG]   bot_lang: {original_bot_lang} → {bot_lang}")
    print(f"[DEBUG]   Kokoro config: lang={kokoro_lang}, voice={kokoro_voice}, bot_lang_name={bot_lang_name}")

    try:
        # ── Transcribe ───────────────────────────────────────
        model = get_whisper_model()
        print(f"[DEBUG] Calling Whisper with language={user_lang}")
        segments, info = model.transcribe(
            tmp_path,
            language=user_lang,
            beam_size=1,          # greedy – 2-3× faster than default beam_size=5
            vad_filter=True,      # strip leading/trailing silence before decoding
            vad_parameters={"min_silence_duration_ms": 300},
        )
        user_text = " ".join(s.text.strip() for s in segments).strip()
        print(f"[DEBUG] Whisper detected language: {info.language}, Transcription: {user_text[:50]}...")

        if not user_text:
            raise HTTPException(
                status_code=422,
                detail="No speech detected. Please speak clearly and try again."
            )

        # ── LLM reply ────────────────────────────────────────
        print(f"[DEBUG] Sending LLM request in {bot_lang_name}")
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
        print(f"[DEBUG] LLM replied: {bot_text[:50]}...")

        if not bot_text:
            raise HTTPException(status_code=500, detail="LLM returned an empty response.")

        # ── TTS ──────────────────────────────────────────────
        print(f"[DEBUG] Synthesizing with Kokoro: lang={kokoro_lang}, voice={kokoro_voice}")
        ts = time.strftime("%Y%m%d_%H%M%S")
        audio_filename = f"response_{ts}_{uuid.uuid4().hex[:6]}.wav"
        audio_path = os.path.join(OUTPUT_FOLDER, audio_filename)

        kokoro = get_kokoro_model()
        samples, sample_rate = kokoro.create(bot_text, voice=kokoro_voice, speed=1.0, lang=kokoro_lang)
        sf.write(audio_path, samples, sample_rate)
        print(f"[DEBUG] TTS complete, saved to: {audio_filename}")

        # ── Store & return ───────────────────────────────────
        entry = {
            "id": str(uuid.uuid4()),
            "user": user_text,
            "bot": bot_text,
            "audio_url": f"/audio/{audio_filename}",
            "timestamp": time.strftime("%H:%M"),
        }
        _chat_history[sid].append(entry)
        return JSONResponse(entry)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    # Prevent path traversal attacks
    safe_name = os.path.basename(filename)
    path = os.path.join(OUTPUT_FOLDER, safe_name)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(path, media_type="audio/wav")


@app.get("/history")
async def history(request: Request):
    sid = request.session.get("sid", "")
    return JSONResponse(_chat_history.get(sid, []))


@app.post("/new_chat")
async def new_chat(request: Request):
    sid = request.session.get("sid", "")
    if sid and sid in _chat_history:
        _chat_history[sid] = []
    return JSONResponse({"ok": True})


# Startup event
@app.on_event("startup")
async def startup_event():
    print("Warming up models (this may take a moment on first run)…")
    try:
        get_whisper_model()
        get_kokoro_model()
    except Exception as exc:
        print(f"[WARN] Model pre-load failed: {exc}")

    print(f"\nServer ready →  http://localhost:{FASTAPI_PORT}\n")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=FASTAPI_PORT)
