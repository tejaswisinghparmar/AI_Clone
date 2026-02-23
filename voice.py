"""
voice.py — TTS Engine (Qwen3-TTS 0.6B-Base)
=============================================
Zero-shot voice cloning via the `qwen-tts` package.

Model : Qwen/Qwen3-TTS-12Hz-0.6B-Base
Features:
  • Pre-computed voice-clone prompt (avoids re-extracting features per call)
  • Sentence-by-sentence streaming synthesis
  • GPU load/unload for VRAM-constrained laptops
  • Automatic reference-audio transcription via Whisper fallback
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from threading import Lock
from typing import AsyncGenerator

import time

import numpy as np
import sounddevice as sd
import torch

logger = logging.getLogger(__name__)

# ─── Configuration ──────────────────────────────────────────────────────────────
TTS_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
DEFAULT_LANGUAGE = "Auto"


def _resolve_attn_impl() -> str:
    """Pick the best available attention implementation."""
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        logger.info("flash-attn not installed → falling back to SDPA (slightly slower)")
        return "sdpa"


def _clean_for_speech(text: str) -> str:
    """Strip Markdown / formatting artefacts so the TTS reads natural prose."""
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"`+", "", text)
    text = re.sub(r"#{1,6}\s*", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", "", text)
    text = re.sub(r"[-*+]\s", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ─── Main Class ─────────────────────────────────────────────────────────────────

class VoiceSynthesizer:
    """
    Qwen3-TTS zero-shot voice-cloning TTS engine.

    Lifecycle managed by the GUI:
        voice.load()                       # GPU ← weights
        audio, sr = voice.synthesize(text)  # generate speech
        voice.unload()                     # GPU → free  (optional)
    """

    def __init__(
        self,
        model_id: str = TTS_MODEL_ID,
        ref_audio_path: str = "ref_audio.wav",
        ref_text: str = "",
        language: str = DEFAULT_LANGUAGE,
        device: str = "cuda:0",
        **kwargs,
    ):
        self.model_id = model_id
        self.ref_audio_path = Path(ref_audio_path)
        self.ref_text = ref_text
        self.language = language
        self.device = device

        self._model = None
        self._voice_prompt = None
        self._sample_rate: int | None = None
        self._lock = Lock()
        self._loaded = False

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ── Model lifecycle ─────────────────────────────────────────────────────

    def load(self) -> None:
        """Load model onto GPU and pre-compute voice-clone prompt."""
        if self._loaded:
            return

        with self._lock:
            if self._loaded:
                return

            from qwen_tts import Qwen3TTSModel

            attn = _resolve_attn_impl()
            logger.info("Loading TTS model: %s  (attn=%s)", self.model_id, attn)

            self._model = Qwen3TTSModel.from_pretrained(
                self.model_id,
                device_map=self.device,
                dtype=torch.bfloat16,
                attn_implementation=attn,
            )

            self._prepare_voice_prompt()
            self._loaded = True
            logger.info("✅ TTS engine ready")

    def unload(self) -> None:
        """Free GPU VRAM occupied by the TTS model."""
        if not self._loaded:
            return

        with self._lock:
            del self._model
            self._model = None
            self._voice_prompt = None
            self._loaded = False

            torch.cuda.empty_cache()
            logger.info("TTS model unloaded — VRAM freed")

    # ── Voice-clone prompt ──────────────────────────────────────────────────

    def _prepare_voice_prompt(self) -> None:
        """
        Pre-compute the voice-clone conditioning from ref_audio.wav.
        Cached and reused for every synthesis call.
        """
        if not self.ref_audio_path.exists():
            logger.warning(
                "⚠️  Reference audio not found: %s\n"
                "   → Voice cloning DISABLED.",
                self.ref_audio_path,
            )
            self._voice_prompt = None
            return

        ref_audio_str = str(self.ref_audio_path)

        if self.ref_text:
            logger.info("Creating voice-clone prompt (ICL mode — transcript provided)")
            self._voice_prompt = self._model.create_voice_clone_prompt(
                ref_audio=ref_audio_str,
                ref_text=self.ref_text,
                x_vector_only_mode=False,
            )
        else:
            logger.info("Creating voice-clone prompt (x-vector mode — no transcript)")
            self._voice_prompt = self._model.create_voice_clone_prompt(
                ref_audio=ref_audio_str,
                ref_text="",
                x_vector_only_mode=True,
            )

    def auto_transcribe_reference(self, whisper_model) -> None:
        """
        Use a loaded faster-whisper model to auto-generate the ref_text
        transcript, upgrading clone quality.
        """
        if self.ref_text or not self.ref_audio_path.exists():
            return

        import torchaudio

        waveform, sr = torchaudio.load(str(self.ref_audio_path))
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        audio_np = waveform.squeeze(0).numpy()

        segments, _ = whisper_model.transcribe(audio_np, beam_size=5)
        self.ref_text = " ".join(seg.text for seg in segments).strip()
        logger.info("Auto-transcribed reference audio: '%s'", self.ref_text)

    # ── Synthesis ───────────────────────────────────────────────────────────

    @torch.inference_mode()
    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """
        Synthesize speech from *text* using the cloned voice.

        Returns (audio_float32, sample_rate).
        """
        if not self._loaded:
            self.load()

        text = _clean_for_speech(text)
        if not text:
            return np.array([], dtype=np.float32), 24_000

        voice_prompt = self._voice_prompt
        if voice_prompt is None:
            logger.warning("No voice-clone prompt — skipping TTS for: %s", text[:60])
            return np.array([], dtype=np.float32), 24_000

        t0 = time.perf_counter()
        wavs, sr = self._model.generate_voice_clone(
            text=text,
            language=self.language,
            voice_clone_prompt=voice_prompt,
        )
        dt = time.perf_counter() - t0
        logger.info("TTS synthesised %d chars in %.1f s", len(text), dt)

        audio = wavs[0] if isinstance(wavs, list) else wavs
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()

        self._sample_rate = sr
        return audio.astype(np.float32), sr

    # ── Playback ────────────────────────────────────────────────────────────

    @staticmethod
    def play_audio(audio: np.ndarray, sr: int, blocking: bool = True) -> None:
        if audio.size == 0:
            return
        sd.play(audio, samplerate=sr)
        if blocking:
            sd.wait()

    @staticmethod
    def wait_audio() -> None:
        sd.wait()

    # ── Async helpers ───────────────────────────────────────────────────────

    async def speak(self, text: str, wait: bool = True) -> None:
        loop = asyncio.get_event_loop()
        audio, sr = await loop.run_in_executor(None, self.synthesize, text)
        if audio.size > 0:
            if wait:
                await loop.run_in_executor(None, self.play_audio, audio, sr)
            else:
                await loop.run_in_executor(None, self.play_audio, audio, sr, False)

    async def stream_and_speak(
        self,
        text_stream: AsyncGenerator[str, None],
        cancel_event: asyncio.Event | None = None,
    ) -> None:
        """
        Consume an async stream of text tokens (from the LLM), buffer
        until a sentence boundary, then synthesize + play each sentence.
        Stops early if cancel_event is set.
        """
        loop = asyncio.get_event_loop()
        buffer = ""
        sentence_enders = frozenset(".!?\n")
        pending_audio: tuple[np.ndarray, int] | None = None
        tts_batch = ""
        MIN_BATCH_CHARS = 40

        async def _flush_batch(text: str) -> None:
            nonlocal pending_audio
            if cancel_event and cancel_event.is_set():
                return
            audio, sr = await loop.run_in_executor(None, self.synthesize, text)
            if pending_audio is not None:
                await loop.run_in_executor(None, self.wait_audio)
            if cancel_event and cancel_event.is_set():
                return
            if audio.size > 0:
                await loop.run_in_executor(None, self.play_audio, audio, sr, False)
                pending_audio = (audio, sr)
            else:
                pending_audio = None

        async for chunk in text_stream:
            if cancel_event and cancel_event.is_set():
                break
            buffer += chunk
            while True:
                pos = next(
                    (i for i, ch in enumerate(buffer) if ch in sentence_enders),
                    -1,
                )
                if pos == -1:
                    break
                sentence = buffer[: pos + 1].strip()
                buffer = buffer[pos + 1 :]
                if sentence:
                    tts_batch += (" " if tts_batch else "") + sentence
                    if len(tts_batch) >= MIN_BATCH_CHARS or "\n" in sentence:
                        await _flush_batch(tts_batch)
                        tts_batch = ""

        if not (cancel_event and cancel_event.is_set()):
            remainder = buffer.strip()
            if remainder:
                tts_batch += (" " if tts_batch else "") + remainder
            if tts_batch.strip():
                await _flush_batch(tts_batch.strip())
            if pending_audio is not None:
                await loop.run_in_executor(None, self.wait_audio)

    # ── Diagnostics ─────────────────────────────────────────────────────────

    def get_vram_mb(self) -> float:
        return 1_200.0 if self._loaded else 0.0


# ─── Edge TTS (network-based, zero GPU) ────────────────────────────────────────

class EdgeTTSSynthesizer:
    """
    Lightweight network-based TTS via Microsoft Edge Read Aloud.
    Zero GPU cost, ~0.5–1.5 s latency, decent narrator voice.
    """

    # Good English voices — pick one with a natural cadence
    DEFAULT_VOICE = "en-US-GuyNeural"

    def __init__(self, voice: str = DEFAULT_VOICE):
        self.voice = voice
        self._loaded = True   # no model to load

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        pass  # nothing to load

    def unload(self) -> None:
        pass  # nothing to unload

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Synthesize speech using Edge TTS.  Returns (audio_f32, 24000)."""
        import asyncio
        import edge_tts
        import io
        import soundfile as sf

        text = _clean_for_speech(text)
        if not text:
            return np.array([], dtype=np.float32), 24_000

        async def _generate() -> bytes:
            comm = edge_tts.Communicate(text, self.voice)
            chunks = []
            async for chunk in comm.stream():
                if chunk["type"] == "audio":
                    chunks.append(chunk["data"])
            return b"".join(chunks)

        t0 = time.time()

        # Run in a new loop to avoid nesting issues
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We're inside an async context — run in a thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                raw = pool.submit(lambda: asyncio.run(_generate())).result()
        else:
            raw = asyncio.run(_generate())

        dt = time.time() - t0
        logger.info("Edge TTS synthesised %d chars in %.1f s", len(text), dt)

        # Decode MP3 → float32 numpy
        audio, sr = sf.read(io.BytesIO(raw), dtype="float32")
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)  # stereo → mono

        return audio.astype(np.float32), sr

    async def stream_and_speak(
        self,
        text_stream,
        cancel_event: asyncio.Event | None = None,
    ) -> None:
        """Edge TTS version of stream_and_speak with cancel support."""
        loop = asyncio.get_event_loop()
        buffer = ""
        sentence_enders = frozenset(".!?\n")
        pending = False
        tts_batch = ""
        MIN_BATCH = 50

        async def _flush(text: str) -> None:
            nonlocal pending
            if cancel_event and cancel_event.is_set():
                return
            audio, sr = await loop.run_in_executor(None, self.synthesize, text)
            if pending:
                await loop.run_in_executor(None, sd.wait)
            if cancel_event and cancel_event.is_set():
                return
            if audio.size > 0:
                sd.play(audio, samplerate=sr)
                pending = True
            else:
                pending = False

        async for chunk in text_stream:
            if cancel_event and cancel_event.is_set():
                break
            buffer += chunk
            while True:
                pos = next(
                    (i for i, ch in enumerate(buffer) if ch in sentence_enders), -1)
                if pos == -1:
                    break
                sentence = buffer[: pos + 1].strip()
                buffer = buffer[pos + 1 :]
                if sentence:
                    tts_batch += (" " if tts_batch else "") + sentence
                    if len(tts_batch) >= MIN_BATCH or "\n" in sentence:
                        await _flush(tts_batch)
                        tts_batch = ""

        remainder = buffer.strip()
        if remainder:
            tts_batch += (" " if tts_batch else "") + remainder
        if tts_batch.strip():
            await _flush(tts_batch.strip())
        if pending:
            await loop.run_in_executor(None, sd.wait)
