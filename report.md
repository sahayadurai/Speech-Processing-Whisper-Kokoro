# Speech Chatbot Web App — Project Report

**Student:** Sahaya Gnanadurai [D03000149]
**Course:** Speech Processing
**Assignment:** Web Interface for Voice Chatbot
**Framework:** FastAPI · Browser-based recording · Minimalist UI

---

## Overview

A real-time, browser-based voice chatbot built with FastAPI. Users hold the spacebar (or click) in the web UI to record speech, which is streamed to a FastAPI backend, transcribed with faster-whisper, answered by an Ollama LLM, synthesised to speech with Kokoro TTS, and played back in the browser. Chat history is maintained per session. The UI is minimalist with a gradient background, responsive layout, and intuitive controls for multilingual STT and TTS (English, Italian, French; German for STT only).

---

## Frontend Design

### Technologies
| Layer | Library / Approach |
|---|---|
| Responsive layout | Vanilla CSS Grid + Flexbox |
| Styling | Inline CSS with gradient backgrounds |
| Fonts | System default (`-apple-system`, `Segoe UI`, etc.) |
| Recording | Web MediaRecorder API (`audio/webm`) |
| Audio playback | HTML5 `Audio` object |
| Animations | CSS keyframes (fade-in, pulse) |

### Minimalist UI Features
- **Gradient Header**: Purple gradient background with app title and hint
- **Control Panel**: Language selection dropdowns (Listen/Reply) in a clean, compact layout
- **Chat Display**: Scrollable message area with user messages on the right (blue), bot messages on the left (gray), timestamps, and embedded audio players
- **Record Button**: Large, prominent push-to-talk button with visual states:
  - Default: Blue "🎤 Record"
  - Recording: Red "⏹ Recording..." with pulse animation
  - Processing: Gray "⏳ Processing..." (disabled)
- **Spacebar Support**: Hold spacebar to record; release to stop. Focus-aware to avoid accidental activation in inputs
- **New Chat Button**: Green button to clear history and start fresh
- **Status Display**: Real-time feedback (Processing, Done, Errors)
- **Responsive Design**: Adapts to mobile devices with touch support

### Interaction Modes
1. **Mouse/Trackpad**: `pointerdown` → record, `pointerup` → stop
2. **Touch**: `touchstart` → record, `touchend` → stop
3. **Keyboard**: `spacebar down` → record, `spacebar up` → stop (disabled during active element focus)

---

## Backend (app.py) — FastAPI

### Framework Advantages
- **Async-ready**: Native support for async operations, reducing latency
- **Type hints**: Full type annotation support improves code clarity
- **Auto-documentation**: Built-in OpenAPI docs at `/docs`
- **Middleware**: Simple session and CORS management via Starlette
- **Performance**: Lighter weight than Flask with faster request handling

### Routes

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Serve `index.html`; create session if new |
| `POST` | `/chat` | Receive audio blob → STT → LLM → TTS → return JSON |
| `POST` | `/new_chat` | Clear server-side chat history for current session |
| `GET` | `/audio/<filename>` | Serve WAV file (path-traversal safe) |
| `GET` | `/history` | Return current session's message list as JSON |

### Session Management
- Uses Starlette `SessionMiddleware` with HTTP-only cookies
- Each browser tab/window gets a UUID session ID (`sid`)
- `_chat_history` dict maps `sid` → list of message dicts
- History survives page refresh (session cookie + server-side dict)
- History is lost on server restart (in-memory only — sufficient for local PoC)

### `/chat` Pipeline Optimisations

| Step | Technique | Benefit |
|---|---|---|
| Transcription | `beam_size=1` (greedy decoding) | ~2-3× faster than default `beam_size=5` |
| | `vad_filter=True` | Strips silence before decoding; reduces wasted compute |
| | `vad_parameters` min_silence 300ms | Filters brief pauses, improves latency |
| LLM Response | `options.num_predict: 80` | Caps output tokens; sufficient for 1-2 sentences |
| | Early exit on empty | Prevents unnecessary TTS pipeline |

### Thread Safety
- `_init_lock` (threading.Lock) protects model initialisation
- Models are loaded once and reused across all requests
- FastAPI runs with multiple workers via Uvicorn, each with own model instances

---

## Automated Setup (run.sh)

The `run.sh` script automates all steps in a single click:

| Step | Action | Details |
|---|---|---|
| 1 | Install Homebrew | Checks / installs package manager if missing |
| 2 | Install ffmpeg | Required by `faster-whisper` for audio decoding |
| 3 | Install Ollama | Local LLM inference engine |
| 4 | Start Ollama service | Runs in background; polls readiness (15s timeout) |
| 5 | Pull LLM model | Downloads `qwen3:1.7b` (~1.4 GB) if missing |
| 6 | Create Python venv | Isolated environment for dependencies |
| 7 | Install Python packages | All packages from `requirements.txt` (FastAPI, uvicorn, etc.) |
| 8 | Download Kokoro models | Kokoro ONNX (~310 MB) + voices file (~27 MB) |
| 9 | Launch FastAPI app | Starts `uvicorn` server on configured port |

**One-line to start:**
```bash
./run.sh
```

---

## Dependencies

| Category | Package | Purpose |
|---|---|---|
| **Framework** | FastAPI | Web framework (replaces Flask) |
| | Uvicorn | ASGI server (runs FastAPI) |
| | Starlette | ASGI middleware (sessions, CORS) |
| **Audio** | faster-whisper | Speech-to-text (faster variant of OpenAI Whisper) |
| | kokoro-onnx | Text-to-speech (local, no external API) |
| | soundfile | WAV file writing |
| **HTTP** | requests | Ollama API communication |
| **Config** | python-dotenv | Environment variable management |
| **Numeric** | numpy | Numerical operations |

---

## Environment Variables

**Default values** (can be overridden in `.env`):

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `8080` | FastAPI server port |
| `SECRET_KEY` | `sp-a2-chatbot-local-2026` | Session encryption key |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `OLLAMA_MODEL` | `qwen3:1.7b` | LLM model to use |
| `WHISPER_MODEL` | `turbo` | Whisper variant (turbo, base, small, etc.) |
| `KOKORO_VOICE` | `af_sarah` | Default TTS voice |

---

## Key Improvements

1. **FastAPI Migration**
   - Replaced Flask with FastAPI for better async support and performance
   - Cleaner route definitions with type hints
   - Automatic OpenAPI documentation

2. **Minimalist UI**
   - Removed heavy dependencies (Bootstrap, Tailwind CDN)
   - Inline CSS for smaller bundle and faster load
   - Gradient design for modern aesthetic
   - Touch and spacebar support for accessibility

3. **Streamlined Setup**
   - Single `run.sh` script automates everything
   - Removed duplicate code and outdated branches
   - Cleaner output and error messages

4. **Better Documentation**
   - Clear dependency tables
   - Step-by-step setup explanation
   - Environment variable reference

---
