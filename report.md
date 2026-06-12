# Speech Chatbot Web App — Project Report

**Student:** Sahaya Gnanadurai [D03000149]
**Course:** Speech Processing
**Assignment:** Web Interface for Voice Chatbot
**Techstack:** Flask · Browser-based recording

---

## Overview

A real-time, browser-based push-to-talk voice chatbot. The user holds a button (or the spacebar) in the web UI to record speech, which is streamed to a Flask backend, transcribed with faster-whisper, answered by an Ollama LLM, synthesised to speech with Kokoro TTS, and played back in the browser. Chat history is maintained per session and visible in a toggleable sidebar. The UI supports multilingual STT and TTS (English, Italian, French; German for STT only), a "New Chat" button.

---

## Frontend Design

### Technologies
| Layer | Library / Approach |
|---|---|
| Responsive layout | Bootstrap 5 (CDN) |
| Utility styling & dark mode | Tailwind CSS (CDN, `darkMode: 'class'`) |
| Fonts | System default (`-apple-system`, `Segoe UI`, etc.) |
| Recording | Web MediaRecorder API (`audio/webm`) |
| Audio playback | HTML5 `Audio` object |
| Background animation | GSAP 3 ticker + Canvas 2D API |

### Push-to-Talk Button
- Uses `pointerdown` / `pointerup` events — works for both mouse and touch
- `setPointerCapture` ensures `pointerup` fires even if the pointer leaves the button
- `touch-action: none` disables browser scroll interference
- Visual states: default → red "● Recording" fill → "Processing…" (disabled) → default
- **Spacebar hold**: `keydown`/`keyup` listeners on `Space` trigger the same record/stop flow; focus on interactive elements (input, select, button, textarea) suppresses it to prevent accidental activation

### Sidebar History
- Lists all exchanges in the current session, newest at the bottom
- Each item shows timestamp + first 50 characters of user speech
- Click scrolls the main chat to that exchange and closes the sidebar
- Shows "No messages yet." placeholder when empty

---

## Backend (app.py)

### Routes

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Serve `index.html`; create Flask session if new |
| `POST` | `/chat` | Receive audio blob → STT → LLM → TTS → return JSON |
| `POST` | `/new_chat` | Clear server-side chat history for the current session |
| `GET` | `/audio/<filename>` | Serve WAV file (path-traversal safe) |
| `GET` | `/history` | Return current session's message list as JSON |

### `/chat` Optimisations
- `beam_size=1` — greedy Whisper decoding, ~2-3× faster than default `beam_size=5`
- `vad_filter=True` — silence trimmed before sending audio to Whisper, reduces wasted decode time
- `options.num_predict: 80` — caps Ollama output tokens; sufficient for 1-2 spoken sentences, reduces LLM latency

### `/chat` Pipeline

### Session Management
- Each browser tab gets a Flask session cookie containing a UUID (`sid`)
- `_chat_history` dict maps `sid` → list of message dicts
- History survives page refresh (session cookie + server-side dict)
- History is lost on server restart (in-memory only — sufficient for local PoC)

### Thread Safety
- `_init_lock` (threading.Lock) protects model initialisation
- Models are loaded once and reused across all requests
- Flask runs in threaded mode (`threaded=True`)

---

### Step-by-step Description

| Step | What it does |
|------|-------------|
| 1 | Checks / installs **Homebrew** |
| 2 | Checks / installs **ffmpeg** via brew — required by `faster-whisper` |
| 3 | Checks / installs **Ollama** via brew |
| 4 | Starts `ollama serve` in background if not already running; polls readiness up to 15 s |
| 5 | Pulls `qwen3:1.7b` (~1.4 GB) if not already present in Ollama |
| 6 | Creates Python `.venv` if it does not exist |
| 7 | Activates the venv and installs / updates all packages from `requirements.txt` |
| 8 | Removes stale `kokoro-v0_19.onnx` / `voices.bin` if present |
| 9 | Downloads `kokoro-v1.0.onnx` (~310 MB) if missing or under 50 MB (corrupt) |
| 10 | Downloads `voices-v1.0.bin` (~27 MB) if missing |
| 11 | Launches `app.py` and prints the browser URL |

---
