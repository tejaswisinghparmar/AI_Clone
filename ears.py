"""
ears.py — Speech-to-Text Engine
================================
Live microphone → text using faster-whisper + Silero VAD + PyAudio.

Model  : faster-whisper  tiny.en          (CUDA, int8_float16)
VAD    : Silero VAD v5  (~1 MB, CPU-only)
Audio  : PyAudio  16 kHz / mono / float32

Pipeline
--------
1. PyAudio opens a 16 kHz mic stream.
2. 32 ms chunks are fed through Silero VAD.
3. State machine: IDLE → SPEAKING → TRAILING_SILENCE → transcribe.
4. The accumulated speech buffer is sent to faster-whisper.
5. Transcription text is returned.
"""

from __future__ import annotations

import asyncio
import gc
import logging
from typing import Optional

import numpy as np
import pyaudio
import torch
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

# ─── Configuration ──────────────────────────────────────────────────────────────
WHISPER_MODEL_SIZE = "tiny.en"
COMPUTE_TYPE       = "int8_float16"   # ~40 % less VRAM than pure float16

# Fallbacks for low-VRAM GPUs
WHISPER_FALLBACK_MODELS = [
    "tiny.en",
    "tiny",
]
CPU_FALLBACK_MODEL = "tiny.en"
CPU_COMPUTE_TYPE = "int8"

SAMPLE_RATE        = 16_000           # Hz  (required by both Whisper & Silero)
CHANNELS           = 1
CHUNK_DURATION_MS  = 32               # Silero VAD window (16 kHz × 32 ms = 512 samples)
CHUNK_SIZE         = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)   # 512

# VAD tuning
VAD_THRESHOLD              = 0.45     # speech probability threshold
SILENCE_AFTER_SPEECH_MS    = 700      # silence needed to finalise an utterance
MIN_SPEECH_DURATION_MS     = 300      # ignore bursts shorter than this
MAX_RECORDING_SECONDS      = 20       # safety cap to prevent runaway recordings


class SpeechListener:
    """
    Live speech-to-text with endpoint detection.

    Typical lifecycle (managed by the orchestrator):
        ears.load()                  # GPU ← Whisper weights
        text = await ears.listen()   # block until user finishes speaking
        ears.unload()                # GPU → free
    """

    def __init__(
        self,
        device: str = "cuda",
        compute_type: str = COMPUTE_TYPE,
    ):
        self.device = device
        self.compute_type = compute_type

        self.model_size = WHISPER_MODEL_SIZE

        self._whisper: Optional[WhisperModel] = None
        self._vad = None
        self._pa: Optional[pyaudio.PyAudio] = None
        self._loaded = False

    # ── Properties ──────────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def whisper(self) -> Optional[WhisperModel]:
        """Expose the raw WhisperModel (used by voice.auto_transcribe_reference)."""
        return self._whisper

    # ── Model lifecycle ─────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load Whisper onto the GPU and initialise Silero VAD (CPU)."""
        if self._loaded:
            return

        models_to_try = [
            WHISPER_MODEL_SIZE,
            *[m for m in WHISPER_FALLBACK_MODELS if m != WHISPER_MODEL_SIZE],
        ]

        last_error: Optional[Exception] = None
        for model_size in models_to_try:
            try:
                logger.info(
                    "Loading Whisper: %s  (compute=%s, device=%s)",
                    model_size, self.compute_type, self.device,
                )
                self._whisper = WhisperModel(
                    model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                )
                self.model_size = model_size
                break
            except RuntimeError as exc:
                last_error = exc
                if "out of memory" in str(exc).lower() and self.device == "cuda":
                    logger.warning(
                        "CUDA OOM while loading '%s'. Trying a smaller model …",
                        model_size,
                    )
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                raise

        if self._whisper is None:
            if self.device == "cuda":
                logger.warning(
                    "All CUDA model sizes failed. Falling back to CPU (%s, %s).",
                    CPU_FALLBACK_MODEL, CPU_COMPUTE_TYPE,
                )
                self.device = "cpu"
                self.compute_type = CPU_COMPUTE_TYPE
                try:
                    self._whisper = WhisperModel(
                        CPU_FALLBACK_MODEL,
                        device=self.device,
                        compute_type=self.compute_type,
                    )
                    self.model_size = CPU_FALLBACK_MODEL
                except Exception as exc:
                    last_error = exc
            if self._whisper is None and last_error is not None:
                raise last_error

        logger.info("Loading Silero VAD …")
        from silero_vad import load_silero_vad
        self._vad = load_silero_vad()

        self._loaded = True
        logger.info(
            "✅ STT engine ready (model=%s, device=%s, compute=%s)",
            self.model_size, self.device, self.compute_type,
        )

    def unload(self) -> None:
        """Free GPU VRAM by deleting the Whisper model (keep tiny VAD)."""
        if not self._loaded:
            return

        if self._whisper is not None:
            del self._whisper
            self._whisper = None

        # Silero VAD is ~1 MB on CPU — keep it alive
        self._loaded = False
        torch.cuda.empty_cache()
        logger.info("Whisper unloaded — VRAM freed")

    # ── PyAudio helpers ─────────────────────────────────────────────────────────

    def _ensure_pyaudio(self) -> pyaudio.PyAudio:
        if self._pa is None:
            self._pa = pyaudio.PyAudio()
        return self._pa

    def _close_pyaudio(self) -> None:
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None

    # ── Core: transcribe a numpy buffer ─────────────────────────────────────────

    def transcribe(self, audio_np: np.ndarray) -> str:
        """Transcribe a float32 numpy waveform (16 kHz) to text."""
        if not self._loaded:
            self.load()

        segments, info = self._whisper.transcribe(
            audio_np,
            beam_size=1,            # greedy decode — much faster on tiny.en
            language="en",          # skip language detection overhead
            vad_filter=True,
            vad_parameters={"threshold": VAD_THRESHOLD},
        )

        text = " ".join(seg.text for seg in segments).strip()
        if text:
            logger.info(
                "Transcribed (%s %.0f%%): %s",
                info.language, info.language_probability * 100, text,
            )
        return text

    # ── Core: live mic → text ───────────────────────────────────────────────────

    def listen_once(self) -> str:
        """
        Block until the user speaks a complete utterance, then return text.

        Uses Silero VAD for real-time endpoint detection:
          IDLE  →  speech detected  →  SPEAKING
          SPEAKING  →  silence > threshold  →  finalise & transcribe
        """
        if not self._loaded:
            self.load()

        pa = self._ensure_pyaudio()
        stream = pa.open(
            format=pyaudio.paFloat32,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
        )

        logger.info("🎤 Listening … (speak now)")

        # Pre-allocate a list; numpy concat at end is faster than per-chunk
        audio_buffer: list[np.ndarray] = []
        buf_samples  = 0
        is_speaking   = False
        silence_chunks = 0
        speech_chunks  = 0

        silence_limit     = int(SILENCE_AFTER_SPEECH_MS / CHUNK_DURATION_MS)
        min_speech_chunks = int(MIN_SPEECH_DURATION_MS / CHUNK_DURATION_MS)
        max_chunks        = int(MAX_RECORDING_SECONDS * 1000 / CHUNK_DURATION_MS)
        total_chunks      = 0
        vad_fn            = self._vad    # local ref avoids dict lookup each iter

        try:
            while total_chunks < max_chunks:
                raw = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                chunk = np.frombuffer(raw, dtype=np.float32).copy()
                total_chunks += 1

                # ── Silero VAD inference (CPU, < 0.1 ms) ────────────────────────
                speech_prob = vad_fn(
                    torch.from_numpy(chunk), SAMPLE_RATE,
                ).item()

                if speech_prob >= VAD_THRESHOLD:
                    # Speech detected
                    if not is_speaking:
                        logger.debug("VAD: speech start")
                    is_speaking = True
                    silence_chunks = 0
                    speech_chunks += 1
                    audio_buffer.append(chunk)

                elif is_speaking:
                    # Silence after speech
                    silence_chunks += 1
                    audio_buffer.append(chunk)       # keep trailing silence

                    if silence_chunks >= silence_limit:
                        if speech_chunks >= min_speech_chunks:
                            logger.debug("VAD: speech end (%.1f s)",
                                         len(audio_buffer) * CHUNK_DURATION_MS / 1000)
                            break
                        else:
                            # Too short — probably a cough / click; reset
                            audio_buffer.clear()
                            is_speaking = False
                            silence_chunks = 0
                            speech_chunks = 0
        finally:
            stream.stop_stream()
            stream.close()
            self._vad.reset_states()

        if not audio_buffer:
            return ""

        audio_np = np.concatenate(audio_buffer)
        logger.info("Captured %.1f s of speech", len(audio_np) / SAMPLE_RATE)
        return self.transcribe(audio_np)

    # ── Async wrapper ───────────────────────────────────────────────────────────

    async def listen(self) -> str:
        """Async version of listen_once (runs mic capture in a thread)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.listen_once)

    # ── Diagnostics ─────────────────────────────────────────────────────────────

    def get_vram_mb(self) -> float:
        """Rough VRAM estimate (MB) for the loaded Whisper model."""
        if not self._loaded:
            return 0.0
        # tiny.en  int8_float16 ≈ 200 MB, float16 ≈ 350 MB (rough)
        return 200.0 if "int8" in self.compute_type else 350.0

    def __del__(self):
        self._close_pyaudio()
