#!/usr/bin/env bash
# ============================================================
# run.sh  —  One-shot setup & launch for sp-a2_chatbot_web (macOS)
# FastAPI-based voice chatbot. Fully automated single-click setup.
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
KOKORO_DIR="$SCRIPT_DIR/models/kokoro"
KOKORO_ONNX="$KOKORO_DIR/kokoro-v1.0.onnx"
KOKORO_VOICES="$KOKORO_DIR/voices-v1.0.bin"
KOKORO_ONNX_URL="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
KOKORO_VOICES_URL="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

# ── Load .env ────────────────────────────────────────────────
if [ -f "$SCRIPT_DIR/.env" ]; then
  set -a
  source <(grep -E '^[A-Z_]+=.*' "$SCRIPT_DIR/.env")
  set +a
fi

# Defaults
PORT="${PORT:-8080}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3:1.7b}"

# ── Colours ──────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ── 1. Homebrew ──────────────────────────────────────────────
info "Checking Homebrew..."
if ! command -v brew &>/dev/null; then
  warn "Homebrew not found. Installing..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi
info "Homebrew OK"

# ── 2. ffmpeg (required by faster-whisper) ───────────────────
info "Checking ffmpeg..."
if ! command -v ffmpeg &>/dev/null; then
  info "Installing ffmpeg..."
  brew install ffmpeg
fi
info "ffmpeg OK"

# ── 3. Ollama ────────────────────────────────────────────────
info "Checking Ollama..."
if ! command -v ollama &>/dev/null; then
  info "Installing Ollama..."
  brew install ollama
fi
info "Ollama OK"

# Start ollama serve in background if not already running
if ! curl -s http://localhost:11434/api/tags &>/dev/null; then
  info "Starting Ollama daemon..."
  ollama serve &>/tmp/ollama.log &
  OLLAMA_PID=$!
  for i in $(seq 1 15); do
    sleep 1
    if curl -s http://localhost:11434/api/tags &>/dev/null; then break; fi
    if [ "$i" -eq 15 ]; then error "Ollama did not start in time. Check /tmp/ollama.log"; fi
  done
  info "Ollama daemon started"
else
  info "Ollama already running"
fi

# Pull the LLM model if not already present
info "Checking qwen3:1.7b model..."
if ! ollama list 2>/dev/null | grep -q "qwen3:1.7b"; then
  info "Pulling qwen3:1.7b (~1.4 GB, this may take a few minutes)..."
  ollama pull qwen3:1.7b
fi
info "qwen3:1.7b ready"

# ── 4. Python virtual environment ───────────────────────────
info "Checking Python virtual environment..."
if [ ! -d "$VENV_DIR" ]; then
  info "Creating virtual environment..."
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
info "Virtual environment active"

# ── 5. Python dependencies ───────────────────────────────────
info "Installing/updating Python packages..."
pip install --upgrade pip --quiet
pip install -r "$SCRIPT_DIR/requirements.txt" --quiet
info "Python packages OK"

# ── 6. Kokoro ONNX model files ───────────────────────────────
mkdir -p "$KOKORO_DIR"

[ -f "$KOKORO_DIR/kokoro-v0_19.onnx" ] && { warn "Removing old kokoro-v0_19.onnx..."; rm "$KOKORO_DIR/kokoro-v0_19.onnx"; }
[ -f "$KOKORO_DIR/voices.bin" ] && { warn "Removing old voices.bin..."; rm "$KOKORO_DIR/voices.bin"; }

if [ ! -f "$KOKORO_ONNX" ] || [ "$(wc -c < "$KOKORO_ONNX")" -lt 52428800 ]; then
  [ -f "$KOKORO_ONNX" ] && rm "$KOKORO_ONNX"
  info "Downloading Kokoro ONNX model (~310 MB)..."
  curl -L --fail --progress-bar -o "$KOKORO_ONNX" "$KOKORO_ONNX_URL" \
    || { rm -f "$KOKORO_ONNX"; error "Failed to download kokoro-v1.0.onnx"; }
fi

if [ ! -f "$KOKORO_VOICES" ]; then
  info "Downloading Kokoro voices file (~27 MB)..."
  curl -L --fail --progress-bar -o "$KOKORO_VOICES" "$KOKORO_VOICES_URL" \
    || { rm -f "$KOKORO_VOICES"; error "Failed to download voices-v1.0.bin"; }
fi

info "Kokoro model files ready"

# ── 7. Launch FastAPI app ────────────────────────────────────
echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  Voice Chatbot (FastAPI) is starting...${NC}"
echo -e "${GREEN}  Open your browser at:  http://localhost:${PORT}${NC}"
echo -e "${GREEN}  Press Ctrl+C to stop.${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""

python -m uvicorn app:app --host 127.0.0.1 --port "$PORT" --reload
