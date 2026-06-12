import os
import time
import threading
import requests
import sounddevice as sd
import soundfile as sf
import numpy as np
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro
from pynput import keyboard as kb

# -----------------------------
# CONFIG
# -----------------------------
SAMPLE_RATE = 16000
CHANNELS = 1

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_AUDIO_FOLDER = os.path.join(BASE_DIR, "input_audio")
MODELS_FOLDER = os.path.join(BASE_DIR, "models")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")
KOKORO_ONNX = os.path.join(MODELS_FOLDER, "kokoro", "kokoro-v1.0.onnx")
KOKORO_VOICES = os.path.join(MODELS_FOLDER, "kokoro", "voices-v1.0.bin")
KOKORO_ONNX_MIN_SIZE = 50 * 1024 * 1024  # 50 MB — a corrupt/partial download is far smaller

LIVE_INPUT = os.path.join(INPUT_AUDIO_FOLDER, "live_input.wav")

os.makedirs(INPUT_AUDIO_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(os.path.join(MODELS_FOLDER, "kokoro"), exist_ok=True)

# Models are loaded once and reused across interactions
_whisper_model = None
_kokoro_model = None


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        print("Loading Whisper model (first run may download ~800 MB)...")
        _whisper_model = WhisperModel(
            "turbo",
            device="cpu",
            compute_type="int8",
            download_root=os.path.join(MODELS_FOLDER, "whisper")
        )
        print("Whisper model ready.")
    return _whisper_model


def get_kokoro_model():
    global _kokoro_model
    if _kokoro_model is None:
        if not os.path.isfile(KOKORO_ONNX):
            raise RuntimeError(
                f"Kokoro ONNX model not found at: {KOKORO_ONNX}\n"
                "Run ./run.sh to download it automatically."
            )
        if os.path.getsize(KOKORO_ONNX) < KOKORO_ONNX_MIN_SIZE:
            os.remove(KOKORO_ONNX)
            raise RuntimeError(
                "Kokoro ONNX model was corrupted (file too small) and has been deleted.\n"
                "Re-run ./run.sh to download it again."
            )
        print("Loading Kokoro TTS model...")
        _kokoro_model = Kokoro(KOKORO_ONNX, KOKORO_VOICES)
        print("Kokoro model ready.")
    return _kokoro_model


# -----------------------------
# HELPERS
# -----------------------------
def get_run_paths():
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    transcript_run = os.path.join(OUTPUT_FOLDER, f"transcript_{timestamp}.txt")
    reply_run = os.path.join(OUTPUT_FOLDER, f"reply_{timestamp}.txt")
    audio_run = os.path.join(OUTPUT_FOLDER, f"response_{timestamp}.wav")

    transcript_latest = os.path.join(OUTPUT_FOLDER, "transcript_latest.txt")
    reply_latest = os.path.join(OUTPUT_FOLDER, "reply_latest.txt")
    audio_latest = os.path.join(OUTPUT_FOLDER, "response_latest.wav")

    return {
        "timestamp": timestamp,
        "transcript_run": transcript_run,
        "reply_run": reply_run,
        "audio_run": audio_run,
        "transcript_latest": transcript_latest,
        "reply_latest": reply_latest,
        "audio_latest": audio_latest,
    }


def record_microphone(output_path: str) -> None:
    print("\nHold SPACE to record...")
    print("Release SPACE to stop recording.")

    audio_frames = []
    recording = threading.Event()
    done = threading.Event()

    def callback(indata, frames, time_info, status):
        if status:
            print(status)
        if recording.is_set():
            audio_frames.append(indata.copy())

    def on_press(key):
        if key == kb.Key.space and not recording.is_set():
            recording.set()
            print("Recording...")

    def on_release(key):
        if key == kb.Key.space:
            recording.clear()
            done.set()
            return False  # Stop listener

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        callback=callback
    ):
        with kb.Listener(on_press=on_press, on_release=on_release) as listener:
            done.wait(timeout=60.0)  # safety: auto-stop after 60 s if release missed
            # do NOT call listener.join() here — on_release returns False which
            # already signals the listener to stop; the context manager handles cleanup

    print("Recording stopped.")

    if not audio_frames:
        raise RuntimeError("No audio was recorded.")

    audio = np.concatenate(audio_frames, axis=0)
    sf.write(output_path, audio, SAMPLE_RATE)
    print(f"Saved microphone input to: {output_path}")


def run_whisper(input_audio_path: str, transcript_target: str, transcript_latest: str) -> str:
    print("\nTranscribing with Whisper...")

    model = get_whisper_model()
    segments, _ = model.transcribe(input_audio_path, language="en")
    user_text = " ".join(seg.text.strip() for seg in segments).strip()

    if not user_text:
        raise RuntimeError("Transcript is empty.")

    for path in (transcript_target, transcript_latest):
        with open(path, "w", encoding="utf-8") as f:
            f.write(user_text)

    return user_text


def generate_reply(user_text: str, reply_target: str, reply_latest: str) -> str:
    print("\nSending to Qwen...")

    payload = {
        "model": "qwen3:1.7b",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a smart, friendly, slightly sarcastic voice assistant. "
                    "Reply naturally in English in 1 or 2 short sentences only. "
                    "Be concise because your answer will be spoken aloud. "
                    "Do not output reasoning, analysis, or hidden thoughts. "
                    "Output only the final spoken answer."
                )
            },
            {
                "role": "user",
                "content": user_text
            }
        ],
        "stream": False,
        "think": False
    }

    response = requests.post(
        "http://localhost:11434/api/chat",
        json=payload,
        timeout=120
    )
    response.raise_for_status()

    data = response.json()
    bot_reply = (data.get("message", {}).get("content", "") or "").strip()

    if not bot_reply:
        raise RuntimeError(f"Qwen returned an empty reply. Raw response: {data}")

    with open(reply_target, "w", encoding="utf-8") as f:
        f.write(bot_reply)

    with open(reply_latest, "w", encoding="utf-8") as f:
        f.write(bot_reply)

    return bot_reply


def synthesize_speech(text: str, audio_target: str, audio_latest: str) -> None:
    print("\nSynthesizing with Kokoro...")

    kokoro = get_kokoro_model()
    samples, sample_rate = kokoro.create(text, voice="af_sarah", speed=1.0, lang="en-us")

    for path in (audio_target, audio_latest):
        sf.write(path, samples, sample_rate)

    print(f"Audio response saved at: {audio_target}")


def play_wav_file(file_path: str) -> None:
    print("\n🔊 Playing response...")
    data, sr = sf.read(file_path, dtype="float32")
    sd.play(data, sr)
    sd.wait()


def ensure_services() -> None:
    try:
        requests.get("http://localhost:11434/api/tags", timeout=5)
    except Exception as e:
        raise RuntimeError(
            "Ollama is not running on localhost:11434. "
            "Start it with: ollama serve"
        ) from e


# -----------------------------
# MAIN LOOP
# -----------------------------
def main():
    print("=== Live Speech Chatbot ===")
    print("Press ENTER to prepare a new interaction.")
    print("Then hold SPACE to record and release SPACE to stop.")
    print("Type q and press ENTER to quit.")

    ensure_services()

    while True:
        choice = input("\nENTER = continue | q = quit : ").strip().lower()

        if choice == "q":
            print("Exiting.")
            break

        paths = get_run_paths()

        try:
            record_microphone(LIVE_INPUT)

            user_text = run_whisper(
                LIVE_INPUT,
                paths["transcript_run"],
                paths["transcript_latest"]
            )
            print("\nUser said:")
            print(user_text)

            bot_reply = generate_reply(
                user_text,
                paths["reply_run"],
                paths["reply_latest"]
            )
            print("\nBot reply:")
            print(bot_reply)

            synthesize_speech(
                bot_reply,
                paths["audio_run"],
                paths["audio_latest"]
            )

            play_wav_file(paths["audio_run"])

            print("\nPipeline completed successfully.")
            print(f"Transcript file: {paths['transcript_run']}")
            print(f"Reply file:      {paths['reply_run']}")
            print(f"Audio file:      {paths['audio_run']}")

        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Try again.")

        time.sleep(0.5)


if __name__ == "__main__":
    main()