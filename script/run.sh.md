
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