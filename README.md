# AI_Clone
AI_Clone is an end-to-end local voice-and-personality clone. It combines live speech-to-text, an on-device LLM via Ollama, and zero-shot TTS to deliver near-instant speech-to-speech interaction on modest GPUs.

## Overview
This project runs a loop of **listen → think → speak**:
- **Ears (STT):** faster-whisper + Silero VAD for live microphone transcription.
- **Brain (LLM):** Ollama streaming chat with a custom model defined in Modelfile.
- **Voice (TTS):** Qwen3-TTS zero-shot voice cloning using a short reference clip.

Two entry points are provided:
- `main.py` for a terminal experience with a VRAM-aware pipeline.
- `gui.py` for a ChatGPT-style desktop app with mic input and streaming output.

## Features
- Real-time mic transcription with endpoint detection.
- Streaming LLM responses, synced to sentence-level TTS.
- Voice cloning from a 3–15 second reference WAV.
- GPU memory management to fit on 4–6 GB VRAM laptops.
- Optional narrator mode via Edge TTS (no GPU usage).

## Requirements
Minimum setup (Linux recommended):
- NVIDIA GPU with CUDA (tested on RTX 3050 Laptop).
- Python 3.10+ (3.12 recommended).
- Ollama installed and running.
- PortAudio / PyAudio (for microphone input).

See [requirements.txt](requirements.txt) for Python dependencies.

## Setup
1. Clone and enter the project:
	```bash
	git clone <your-repo-url>
	cd AI_Clone-main
	```

2. Install Python deps:
	```bash
	python -m venv .venv
	source .venv/bin/activate
	pip install -r requirements.txt
	```

3. Install and start Ollama:
	```bash
	sudo systemctl start ollama
	```

4. Create your LLM model from the Modelfile:
	```bash
	ollama create my-ai-clone -f Modelfile
	ollama list
	```

5. Add a voice reference clip (recommended):
	- Place a 3–15 second WAV file named `ref_audio.wav` in the project root.
	- For best quality, also supply the transcript in `main.py` (optional).

## Run
Terminal mode (VRAM-aware):
```bash
python main.py
```

Desktop GUI:
```bash
python gui.py
```

## Configuration
Key options in `main.py`:
- `AGGRESSIVE_VRAM`: Swap STT/TTS models between turns for low VRAM.
- `REF_AUDIO_PATH`: Path to your reference WAV.
- `REF_AUDIO_TRANSCRIPT`: Optional transcript for better cloning.

## Troubleshooting
- **Ollama not found:** ensure `ollama create my-ai-clone -f Modelfile` ran.
- **No CUDA GPU detected:** this project requires an NVIDIA GPU for TTS.
- **Mic not working:** verify PortAudio/PyAudio is installed and the input device is available.
- **Slow first response:** initial model loads can take a few seconds; subsequent turns are faster.

## Project Structure
- `main.py`: Async orchestrator for listen → think → speak.
- `gui.py`: Desktop UI (CustomTkinter) with streaming and mic control.
- `brain.py`: Ollama client and conversation memory.
- `ears.py`: STT engine (faster-whisper + Silero VAD).
- `voice.py`: TTS engine (Qwen3-TTS + optional Edge TTS).


