"""
main.py — AI Clone Orchestrator
=================================
Coordinates the three subsystems (ears → brain → voice) in an
async listen-think-speak pipeline with smart GPU memory management
for VRAM-constrained laptops (HP Victus, 4–6 GB VRAM).

VRAM strategy
-------------
• Ollama runs out-of-process and manages its own VRAM.
• Only ONE of {Whisper STT, Qwen3 TTS} is loaded at a time.
•  listen phase  → Whisper loaded, TTS unloaded
•  speak  phase  → TTS loaded, Whisper unloaded
• A single swap takes ~2–3 s (acceptable between turns).

Usage
-----
    1. systemctl start ollama
    2. ollama create my-ai-clone -f Modelfile
    3. python main.py
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
import time
from pathlib import Path

import torch

from brain import Brain
from ears import SpeechListener
from voice import VoiceSynthesizer

# ─── User Configuration ────────────────────────────────────────────────────────

LOG_LEVEL = logging.INFO

# Set to True on GPUs with < 4 GB VRAM (swaps Whisper ↔ TTS between turns)
# With tiny.en (~0.2 GB) + Qwen3-TTS (~1.2 GB) both fit on a 3.7 GB GPU.
AGGRESSIVE_VRAM = False

# Path to your voice reference recording (3–15 s, WAV, mono preferred)
REF_AUDIO_PATH = "ref_audio.wav"

# Transcript of ref_audio.wav — leave empty to auto-transcribe via Whisper
#   (providing it manually gives the BEST clone quality)
REF_AUDIO_TRANSCRIPT = ""

# ─── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s │ %(name)-12s │ %(levelname)-5s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("orchestrator")


# ─── GPU Memory Manager ────────────────────────────────────────────────────────

class GPUMemoryManager:
    """
    VRAM traffic cop for laptops.

    On an HP Victus (RTX 3050 4 GB / RTX 4050 6 GB / RTX 4060 8 GB):
      • Ollama (gemma3:12b Q4_K_M) claims 4–6 GB via mmap; it handles
        its own CPU ↔ GPU paging.
    • Whisper tiny.en int8  ≈  0.2 GB
      • Qwen3-TTS 0.6B  bfloat16   ≈  1.2 GB

    When AGGRESSIVE_VRAM is True we guarantee at most one of {STT, TTS}
    is resident on the GPU at any moment.
    """

    def __init__(self, aggressive: bool = True):
        self.aggressive = aggressive

    # ── Queries ─────────────────────────────────────────────────────────────────

    @staticmethod
    def free_vram_mb() -> float:
        if not torch.cuda.is_available():
            return 0.0
        free, _ = torch.cuda.mem_get_info()
        return free / (1024 * 1024)

    @staticmethod
    def gpu_summary() -> str:
        if not torch.cuda.is_available():
            return "⚠️  No CUDA GPU detected"
        name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / (1024 ** 3)
        free_gb = torch.cuda.mem_get_info()[0] / (1024 ** 3)
        return f"{name}  —  {total_gb:.1f} GB total, {free_gb:.1f} GB free"

    # ── Swap logic ──────────────────────────────────────────────────────────────

    def prepare_for_listening(
        self, ears: SpeechListener, voice: VoiceSynthesizer,
    ) -> None:
        if self.aggressive and voice.is_loaded:
            logger.info("♻️  VRAM swap → unloading TTS, loading STT")
            voice.unload()
        if not ears.is_loaded:
            ears.load()

    def prepare_for_speaking(
        self, ears: SpeechListener, voice: VoiceSynthesizer,
    ) -> None:
        if self.aggressive and ears.is_loaded:
            logger.info("♻️  VRAM swap → unloading STT, loading TTS")
            ears.unload()
        if not voice.is_loaded:
            voice.load()


# ─── Orchestrator ───────────────────────────────────────────────────────────────

class Orchestrator:
    """
    Main pipeline:  🎤 Listen  →  🧠 Think  →  🔊 Speak  →  repeat
    """

    BANNER = r"""
    ╔══════════════════════════════════════════════════════════╗
    ║          ✦  S I R I U S . A I   v 2 . 0              ║
    ║  Ears : faster-whisper tiny.en          (CPU int8)       ║
    ║  Brain: gemma3:1b via Ollama                           ║
    ║  Voice: Qwen3-TTS 0.6B-Base  (zero-shot clone)         ║
    ╚══════════════════════════════════════════════════════════╝
    """

    def __init__(self) -> None:
        self.ears = SpeechListener(device="cpu", compute_type="int8")
        self.brain = Brain()
        self.voice = VoiceSynthesizer(
            ref_audio_path=REF_AUDIO_PATH,
            ref_text=REF_AUDIO_TRANSCRIPT,
        )
        self.gpu = GPUMemoryManager(aggressive=AGGRESSIVE_VRAM)
        self._running = False

    # ── Preflight ───────────────────────────────────────────────────────────────

    async def _preflight(self) -> bool:
        """Run all startup checks; return True if everything is go."""
        print(self.BANNER)
        logger.info("Preflight checks …")

        # 1. CUDA
        if not torch.cuda.is_available():
            logger.error("❌  CUDA is not available.  An NVIDIA GPU is required.")
            return False
        logger.info("GPU  : %s", GPUMemoryManager.gpu_summary())

        # 2. Ollama
        if not await self.brain.is_alive():
            logger.error(
                "❌  Ollama is not running or model missing.\n"
                "    1. sudo systemctl start ollama\n"
                "    2. ollama create my-ai-clone -f Modelfile\n"
                "    3. ollama list  (verify)"
            )
            return False
        logger.info("LLM  : ✅  Ollama → %s", self.brain.model)

        # 3. Reference audio
        ref = Path(REF_AUDIO_PATH)
        if ref.exists():
            logger.info("Voice: ✅  Reference audio found (%s)", ref.name)
        else:
            logger.warning(
                "Voice: ⚠️  No ref_audio.wav — clone quality will be reduced.\n"
                "       Record a 3–15 s WAV sample of your voice."
            )

        # 4. Auto-transcribe reference if no transcript provided
        if not REF_AUDIO_TRANSCRIPT and ref.exists():
            logger.info("Auto-transcribing reference audio with Whisper …")
            self.ears.load()
            self.voice.auto_transcribe_reference(self.ears.whisper)
            if AGGRESSIVE_VRAM:
                self.ears.unload()

        # 5. Pre-load TTS so the first turn is instant
        if not self.voice.is_loaded:
            logger.info("Voice: Pre-loading TTS model …")
            self.voice.load()

        logger.info("═" * 58)
        logger.info("  ✅  All checks passed — system ready")
        logger.info("═" * 58)
        return True

    # ── Main loop ───────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Enter the listen → think → speak loop."""
        if not await self._preflight():
            sys.exit(1)

        self._running = True
        print("\n🚀 SIRIUS.AI is LIVE.  Say something!  (say 'goodbye' or Ctrl+C to exit)\n")

        try:
            while self._running:
                # ── Phase 1: Listen ─────────────────────────────────────────
                self.gpu.prepare_for_listening(self.ears, self.voice)
                user_text = await self.ears.listen()

                if not user_text:
                    continue

                # Exit commands
                if user_text.strip().lower() in {
                    "exit", "quit", "goodbye", "stop", "shut down",
                }:
                    print("👋 Goodbye!")
                    break

                print(f"\n👤  You  : {user_text}")

                # ── Phase 2: Think + Speak ──────────────────────────────────
                t_start = time.perf_counter()
                self.gpu.prepare_for_speaking(self.ears, self.voice)

                print("🤖  Clone: ", end="", flush=True)

                # Stream LLM tokens → TTS sentences.
                # The tee generator prints each token to the console AS it
                # arrives from Ollama, while also yielding it to the TTS
                # buffer.  Synthesis blocks at sentence boundaries, which
                # naturally pauses the stream so text and speech stay in sync.
                llm_stream = self.brain.think(user_text)

                async def _tee(stream):
                    async for tok in stream:
                        print(tok, end="", flush=True)
                        yield tok
                    print()     # newline after full response

                await self.voice.stream_and_speak(_tee(llm_stream))
                logger.info("⏱  Total response time: %.1f s", time.perf_counter() - t_start)

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted — shutting down …")

        finally:
            self._shutdown()

    # ── Cleanup ─────────────────────────────────────────────────────────────────

    def _shutdown(self) -> None:
        logger.info("Releasing GPU resources …")
        self.ears.unload()
        self.voice.unload()
        torch.cuda.empty_cache()
        logger.info("Goodbye 👋")


# ─── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    # Let Ctrl+C raise KeyboardInterrupt cleanly in asyncio
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    asyncio.run(Orchestrator().run())


if __name__ == "__main__":
    main()
