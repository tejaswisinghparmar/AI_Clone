"""
gui.py — SIRIUS.AI Desktop Interface
======================================
ChatGPT-style dark-theme desktop app built with CustomTkinter.

Features
--------
• Dark theme with ChatGPT-inspired layout
• Mic button with auto-submit on speech end
• Text input box for typed queries
• Audio notification beep when mic activates
• Toggle between audio output / text-only mode
• Streaming LLM response display
• Conversation history sidebar
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
import time
from typing import Optional

# Reduce CUDA memory fragmentation on small GPUs
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import customtkinter as ctk
import numpy as np
import sounddevice as sd
import torch

from brain import Brain
from ears import SpeechListener
from voice import VoiceSynthesizer, EdgeTTSSynthesizer

logger = logging.getLogger("sirius.gui")

# ─── Theme ──────────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

# Colour palette (ChatGPT-inspired)
BG_DARK       = "#0d0d0d"
BG_SIDEBAR    = "#171717"
BG_MAIN       = "#212121"
BG_INPUT      = "#2f2f2f"
BG_USER_MSG   = "#2f2f2f"
BG_BOT_MSG    = "#212121"
FG_PRIMARY    = "#ececec"
FG_SECONDARY  = "#b4b4b4"
FG_DIM        = "#6e6e6e"
ACCENT        = "#10a37f"
ACCENT_HOVER  = "#1a7f64"
MIC_ACTIVE    = "#ef4444"
MIC_IDLE      = "#6e6e6e"
BORDER_COL    = "#383838"
SPEAKING_COL  = "#a78bfa"

FONT_FAMILY   = "Segoe UI"


# ─── Audio Notification ────────────────────────────────────────────────────────

def _play_beep(freq: int = 880, duration_ms: int = 150, volume: float = 0.25) -> None:
    """Play a short two-tone chime to signal mic activation."""
    try:
        sr = 24_000
        n = int(sr * duration_ms / 1000)
        t = np.linspace(0, duration_ms / 1000, n, dtype=np.float32)
        tone = volume * (
            0.6 * np.sin(2 * np.pi * freq * t, dtype=np.float32) +
            0.4 * np.sin(2 * np.pi * (freq * 1.5) * t, dtype=np.float32)
        )
        fade = int(sr * 0.01)
        tone[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
        tone[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
        sd.play(tone, samplerate=sr)
        sd.wait()
    except Exception:
        pass


def _play_mic_off_beep() -> None:
    """Lower-pitched short beep when mic stops."""
    _play_beep(freq=660, duration_ms=100, volume=0.15)


# ─── Async Loop Runner ─────────────────────────────────────────────────────────

class AsyncBridge:
    """Runs an asyncio event loop in a dedicated daemon thread."""

    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def submit(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def stop(self):
        self._loop.call_soon_threadsafe(self._loop.stop)


# ─── Message Bubble Widget ─────────────────────────────────────────────────────

class MessageBubble(ctk.CTkFrame):
    """A single chat message rendered as a full-width row."""

    def __init__(self, parent, role: str, text: str = "", on_replay=None, **kwargs):
        self.role = role
        self._on_replay = on_replay
        is_user = (role == "user")
        bg = BG_USER_MSG if is_user else BG_BOT_MSG

        super().__init__(parent, fg_color=bg, corner_radius=0, **kwargs)

        inner = ctk.CTkFrame(self, fg_color=bg, corner_radius=0)
        inner.pack(fill="x", padx=(60 if is_user else 20, 20 if is_user else 60), pady=12)

        role_text = "You" if is_user else "✦ SIRIUS"
        ctk.CTkLabel(
            inner, text=role_text,
            font=ctk.CTkFont(family=FONT_FAMILY, size=13, weight="bold"),
            text_color=FG_PRIMARY if is_user else ACCENT,
            anchor="w",
        ).pack(fill="x", pady=(0, 4))

        self._text_label = ctk.CTkLabel(
            inner, text=text,
            font=ctk.CTkFont(family=FONT_FAMILY, size=14),
            text_color=FG_PRIMARY, anchor="w", justify="left",
            wraplength=700,
        )
        self._text_label.pack(fill="x")

        # Replay button for assistant messages
        if not is_user:
            actions = ctk.CTkFrame(inner, fg_color="transparent")
            actions.pack(fill="x", pady=(6, 0))
            self._replay_btn = ctk.CTkButton(
                actions, text="▶  Replay",
                font=ctk.CTkFont(family=FONT_FAMILY, size=11),
                fg_color="transparent", hover_color=BG_INPUT,
                text_color=FG_DIM, width=80, height=24, corner_radius=6,
                command=self._do_replay,
            )
            self._replay_btn.pack(side="left")

    def _do_replay(self) -> None:
        text = self._text_label.cget("text")
        if self._on_replay and text:
            self._on_replay(text)

    def set_text(self, text: str) -> None:
        self._text_label.configure(text=text)


# ─── Sidebar Chat Item ─────────────────────────────────────────────────────────

class ChatHistoryItem(ctk.CTkButton):
    def __init__(self, parent, title: str, on_click=None, **kwargs):
        super().__init__(
            parent, text=title,
            font=ctk.CTkFont(family=FONT_FAMILY, size=13),
            fg_color="transparent", text_color=FG_SECONDARY,
            hover_color=BG_INPUT, anchor="w",
            height=36, corner_radius=8,
            command=on_click, **kwargs,
        )


# ─── Main Application ──────────────────────────────────────────────────────────

class SiriusApp(ctk.CTk):
    """SIRIUS.AI — ChatGPT-style desktop interface."""

    def __init__(self):
        super().__init__()

        self.title("SIRIUS.AI")
        self.geometry("1100x750")
        self.minsize(800, 550)
        self.configure(fg_color=BG_DARK)

        # State
        self._brain: Optional[Brain] = None
        self._ears: Optional[SpeechListener] = None
        self._voice: Optional[VoiceSynthesizer] = None
        self._edge_voice: Optional[EdgeTTSSynthesizer] = None
        self._async = AsyncBridge()
        self._is_listening = False
        self._is_processing = False
        self._tts_mode = "my_voice"   # "off" | "my_voice" | "narrator"
        self._cancel_event: asyncio.Event | None = None
        self._current_bot_bubble: Optional[MessageBubble] = None
        self._conversations: list[dict] = []
        self._current_conv_idx = -1
        self._history_buttons: list[ChatHistoryItem] = []
        self._msg_row = 0

        # Init services in background
        self._init_done = False
        threading.Thread(target=self._init_services, daemon=True).start()

        # Build UI then create first conversation
        self._build_ui()
        self._new_conversation()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Service initialisation ──────────────────────────────────────────────
    #
    #  GPU budget (3.68 GB):
    #    Ollama gemma3:1b (separate proc)  ≈  960 MB
    #    Qwen3-TTS 0.6B  (stays loaded)    ≈ 1200 MB
    #    Whisper tiny.en  (CPU — 0 GPU)    ≈    0 MB
    #    Free VRAM                         ≈ 1500 MB
    #
    #  Key: Whisper runs on CPU so TTS can stay loaded permanently.
    #  No VRAM swapping → no 8s reload delay between turns.

    def _init_services(self) -> None:
        try:
            self._brain = Brain()

            # Whisper on CPU — frees GPU VRAM for TTS
            # tiny.en on CPU is still fast (~2s for short speech)
            self._ears = SpeechListener(device="cpu", compute_type="int8")
            self._ears.load()

            # TTS on GPU — stays loaded permanently (no cold starts)
            self._voice = VoiceSynthesizer(
                ref_audio_path="ref_audio.wav", ref_text="")
            self._voice.load()

            # Edge TTS — zero GPU, network-based narrator voice
            self._edge_voice = EdgeTTSSynthesizer()

            # Auto-transcribe ref audio for better clone quality
            if self._ears.whisper and not self._voice.ref_text:
                self._voice.auto_transcribe_reference(self._ears.whisper)
                # Re-create voice prompt with transcript for higher fidelity
                if self._voice.ref_text and self._voice._model:
                    self._voice._prepare_voice_prompt()

            self._init_done = True
            self.after(0, lambda: self._status_label.configure(
                text="● Online", text_color=ACCENT))
            logger.info("SIRIUS services ready")
        except Exception as e:
            logger.error("Init failed: %s", e)
            self.after(0, lambda: self._status_label.configure(
                text="● Error", text_color=MIC_ACTIVE))

    # ── UI Construction ─────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._build_sidebar()
        self._build_main()

    # -- Sidebar --

    def _build_sidebar(self) -> None:
        sb = ctk.CTkFrame(self, fg_color=BG_SIDEBAR, width=260, corner_radius=0)
        sb.grid(row=0, column=0, sticky="nswe")
        sb.grid_propagate(False)
        sb.grid_rowconfigure(2, weight=1)   # history list gets all stretch

        # Logo
        logo = ctk.CTkFrame(sb, fg_color="transparent")
        logo.grid(row=0, column=0, sticky="ew", padx=16, pady=(20, 8))
        ctk.CTkLabel(
            logo, text="✦  SIRIUS.AI",
            font=ctk.CTkFont(family=FONT_FAMILY, size=20, weight="bold"),
            text_color=FG_PRIMARY,
        ).pack(side="left")

        # New Chat
        ctk.CTkButton(
            sb, text="＋  New Chat",
            font=ctk.CTkFont(family=FONT_FAMILY, size=14),
            fg_color=BG_INPUT, hover_color=BORDER_COL,
            text_color=FG_PRIMARY, height=40, corner_radius=10,
            command=self._new_conversation,
        ).grid(row=1, column=0, sticky="ew", padx=12, pady=(8, 16))

        # History
        self._history_frame = ctk.CTkScrollableFrame(
            sb, fg_color="transparent",
            scrollbar_button_color=BG_INPUT,
            scrollbar_button_hover_color=BORDER_COL,
        )
        self._history_frame.grid(row=2, column=0, sticky="nsew", padx=8)

        # ── Voice mode selector ──
        toggles = ctk.CTkFrame(sb, fg_color="transparent")
        toggles.grid(row=3, column=0, sticky="ew", padx=12, pady=(8, 4))

        ctk.CTkLabel(
            toggles, text="Voice Output",
            font=ctk.CTkFont(family=FONT_FAMILY, size=12),
            text_color=FG_SECONDARY,
        ).pack(anchor="w", pady=(0, 4))

        self._tts_seg = ctk.CTkSegmentedButton(
            toggles,
            values=["Mute", "My Voice", "Narrator"],
            font=ctk.CTkFont(family=FONT_FAMILY, size=11),
            selected_color=ACCENT, selected_hover_color=ACCENT_HOVER,
            unselected_color=BG_INPUT, unselected_hover_color=BORDER_COL,
            text_color=FG_PRIMARY, text_color_disabled=FG_DIM,
            corner_radius=8, height=30,
            command=self._on_tts_mode_change,
        )
        self._tts_seg.set("My Voice")
        self._tts_seg.pack(fill="x")

        # Divider
        ctk.CTkFrame(sb, fg_color=BORDER_COL, height=1).grid(
            row=4, column=0, sticky="ew", padx=16, pady=(4, 4))

        # Status
        bottom = ctk.CTkFrame(sb, fg_color="transparent", height=50)
        bottom.grid(row=5, column=0, sticky="ew", padx=16, pady=(4, 16))
        self._status_label = ctk.CTkLabel(
            bottom, text="● Initialising…",
            font=ctk.CTkFont(family=FONT_FAMILY, size=12),
            text_color=FG_DIM,
        )
        self._status_label.pack(side="left")

    # -- Main area --

    def _build_main(self) -> None:
        main = ctk.CTkFrame(self, fg_color=BG_MAIN, corner_radius=0)
        main.grid(row=0, column=1, sticky="nswe")
        main.grid_rowconfigure(0, weight=1)
        main.grid_columnconfigure(0, weight=1)

        # Chat scroll area
        self._chat_scroll = ctk.CTkScrollableFrame(
            main, fg_color=BG_MAIN,
            scrollbar_button_color=BG_INPUT,
            scrollbar_button_hover_color=BORDER_COL,
        )
        self._chat_scroll.grid(row=0, column=0, sticky="nsew")
        self._chat_scroll.grid_columnconfigure(0, weight=1)

        # Input bar container
        bar_wrap = ctk.CTkFrame(main, fg_color=BG_MAIN, corner_radius=0)
        bar_wrap.grid(row=1, column=0, sticky="ew", padx=40, pady=(0, 20))
        bar_wrap.grid_columnconfigure(0, weight=1)

        # Rounded input bar
        bar = ctk.CTkFrame(
            bar_wrap, fg_color=BG_INPUT, corner_radius=22,
            border_width=1, border_color=BORDER_COL,
        )
        bar.grid(row=0, column=0, sticky="ew")
        bar.grid_columnconfigure(0, weight=1)

        # Text entry
        self._input = ctk.CTkTextbox(
            bar, height=44, fg_color="transparent",
            text_color=FG_PRIMARY, border_width=0,
            font=ctk.CTkFont(family=FONT_FAMILY, size=14),
            wrap="word",
            scrollbar_button_color=BG_INPUT,
        )
        self._input.grid(row=0, column=0, sticky="ew", padx=(20, 4), pady=4)
        self._input.bind("<Return>", self._on_enter)
        self._input.bind("<Shift-Return>", lambda e: None)
        self._input.bind("<KeyRelease>", self._auto_resize)

        # Buttons
        btns = ctk.CTkFrame(bar, fg_color="transparent")
        btns.grid(row=0, column=1, padx=(0, 8), pady=8)

        self._mic_btn = ctk.CTkButton(
            btns, text="◉", width=36, height=36, corner_radius=18,
            fg_color="transparent", hover_color=BORDER_COL,
            text_color=MIC_IDLE, font=ctk.CTkFont(size=20),
            command=self._toggle_mic,
        )
        self._mic_btn.pack(side="left", padx=2)

        self._send_btn = ctk.CTkButton(
            btns, text="↑", width=36, height=36, corner_radius=18,
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
            text_color="white", font=ctk.CTkFont(size=18, weight="bold"),
            command=self._on_send,
        )
        self._send_btn.pack(side="left", padx=2)

    # ── Welcome screen ──────────────────────────────────────────────────────

    def _show_welcome(self) -> None:
        self._welcome = ctk.CTkFrame(self._chat_scroll, fg_color="transparent")
        self._welcome.grid(row=0, column=0, pady=(120, 20), sticky="ew")

        ctk.CTkLabel(
            self._welcome, text="✦",
            font=ctk.CTkFont(size=48), text_color=ACCENT,
        ).pack(pady=(0, 8))
        ctk.CTkLabel(
            self._welcome, text="SIRIUS.AI",
            font=ctk.CTkFont(family=FONT_FAMILY, size=32, weight="bold"),
            text_color=FG_PRIMARY,
        ).pack()
        ctk.CTkLabel(
            self._welcome, text="How can I help you today?",
            font=ctk.CTkFont(family=FONT_FAMILY, size=16), text_color=FG_DIM,
        ).pack(pady=(4, 0))

    def _hide_welcome(self) -> None:
        if hasattr(self, "_welcome") and self._welcome and self._welcome.winfo_exists():
            self._welcome.destroy()
            self._welcome = None

    # ── Input handling ──────────────────────────────────────────────────────

    def _auto_resize(self, _=None) -> None:
        txt = self._input.get("1.0", "end-1c")
        lines = max(1, min(txt.count("\n") + 1, 5))
        self._input.configure(height=max(44, lines * 24))

    def _on_enter(self, event) -> str:
        if event.state & 0x1:          # Shift+Enter → newline
            return ""
        if self._is_processing:
            self._interrupt()
        else:
            self._on_send()
        return "break"

    def _on_send(self) -> None:
        text = self._input.get("1.0", "end-1c").strip()
        if not text or self._is_processing:
            return
        self._input.delete("1.0", "end")
        self._input.configure(height=44)
        self._submit(text)

    # ── Mic ─────────────────────────────────────────────────────────────────

    def _toggle_mic(self) -> None:
        if self._is_processing or self._is_listening or not self._init_done:
            return

        self._is_listening = True
        self._mic_btn.configure(text_color=MIC_ACTIVE, fg_color="#3a1f1f")
        self._status_label.configure(text="● Listening…", text_color=MIC_ACTIVE)

        threading.Thread(target=_play_beep, daemon=True).start()
        threading.Thread(target=self._listen_worker, daemon=True).start()

    def _listen_worker(self) -> None:
        try:
            # Whisper stays loaded permanently — instant response
            text = self._ears.listen_once()
        except Exception as e:
            logger.error("Mic error: %s", e)
            text = ""
        self.after(0, self._on_listen_done, text)

    def _on_listen_done(self, text: str) -> None:
        self._is_listening = False
        self._mic_btn.configure(text_color=MIC_IDLE, fg_color="transparent")
        self._status_label.configure(text="● Online", text_color=ACCENT)
        threading.Thread(target=_play_mic_off_beep, daemon=True).start()

        if text and text.strip():
            self._submit(text.strip())

    # ── TTS mode ────────────────────────────────────────────────────────────

    def _on_tts_mode_change(self, value: str) -> None:
        mode_map = {"Mute": "off", "My Voice": "my_voice", "Narrator": "narrator"}
        self._tts_mode = mode_map.get(value, "my_voice")
        logger.info("TTS mode → %s", self._tts_mode)

    # ── Interrupt ───────────────────────────────────────────────────────────────

    def _interrupt(self) -> None:
        """Stop the current LLM stream + TTS playback immediately."""
        if not self._is_processing:
            return
        logger.info("⛔ Interrupt requested")

        # Signal the async stream to stop
        if self._cancel_event:
            self._cancel_event.set()

        # Stop any playing audio right now
        try:
            sd.stop()
        except Exception:
            pass

        # Append " [interrupted]" to the bot bubble
        if self._current_bot_bubble:
            current = self._current_bot_bubble._text_label.cget("text")
            self._current_bot_bubble.set_text(current + " ⸻ *interrupted*")

        # Note: _finish() will be called by the _stream coroutine when it
        # detects the cancel event and exits naturally.
    # ── Replay ──────────────────────────────────────────────────────────────

    def _replay_message(self, text: str) -> None:
        """Re-speak a bot message using the current TTS mode."""
        if self._is_processing or self._tts_mode == "off":
            return
        self._status_label.configure(text="● Speaking…", text_color=SPEAKING_COL)
        threading.Thread(
            target=self._replay_worker, args=(text,), daemon=True,
        ).start()

    def _replay_worker(self, text: str) -> None:
        try:
            if self._tts_mode == "narrator" and self._edge_voice:
                sentences = self._split_sentences(text)
                pending = False
                for sentence in sentences:
                    audio, sr = self._edge_voice.synthesize(sentence)
                    if pending:
                        sd.wait()
                    if audio.size > 0:
                        sd.play(audio, samplerate=sr)
                        pending = True
                if pending:
                    sd.wait()
            else:
                self._speak_sync(text)
        except Exception as e:
            logger.error("Replay error: %s", e)
        self.after(0, lambda: self._status_label.configure(
            text="● Online", text_color=ACCENT))
    # ── Conversations ───────────────────────────────────────────────────────

    def _new_conversation(self) -> None:
        self._conversations.append({"title": "New Chat", "messages": []})
        self._current_conv_idx = len(self._conversations) - 1
        self._msg_row = 0

        for w in self._chat_scroll.winfo_children():
            w.destroy()

        self._show_welcome()

        if self._brain:
            self._brain.clear_history()
        self._update_sidebar()

    def _switch_conversation(self, idx: int) -> None:
        if idx == self._current_conv_idx:
            return
        self._current_conv_idx = idx
        conv = self._conversations[idx]

        for w in self._chat_scroll.winfo_children():
            w.destroy()

        if not conv["messages"]:
            self._show_welcome()
            self._msg_row = 0
        else:
            for i, m in enumerate(conv["messages"]):
                replay_cb = self._replay_message if m["role"] == "assistant" else None
                b = MessageBubble(
                    self._chat_scroll, role=m["role"], text=m["content"],
                    on_replay=replay_cb,
                )
                b.grid(row=i, column=0, sticky="ew")
            self._msg_row = len(conv["messages"])

        if self._brain:
            self._brain.clear_history()
            self._brain._history = [
                {"role": m["role"], "content": m["content"]}
                for m in conv["messages"]
            ]
        self._update_sidebar()

    def _update_sidebar(self) -> None:
        for btn in self._history_buttons:
            btn.destroy()
        self._history_buttons.clear()

        for i, conv in enumerate(reversed(self._conversations)):
            real_idx = len(self._conversations) - 1 - i
            title = conv["title"]
            if len(title) > 28:
                title = title[:28] + "…"

            btn = ChatHistoryItem(
                self._history_frame, title=title,
                on_click=lambda idx=real_idx: self._switch_conversation(idx),
            )
            if real_idx == self._current_conv_idx:
                btn.configure(fg_color=BG_INPUT)
            btn.pack(fill="x", pady=1)
            self._history_buttons.append(btn)

    # ── Core: submit + stream ───────────────────────────────────────────────

    def _submit(self, text: str) -> None:
        if self._is_processing or not self._init_done:
            return

        self._is_processing = True
        self._cancel_event = asyncio.Event()
        # Change send button to interrupt button
        self._send_btn.configure(
            text="■", fg_color=MIC_ACTIVE, hover_color="#b91c1c",
            state="normal", command=self._interrupt,
        )
        self._hide_welcome()

        conv = self._conversations[self._current_conv_idx]

        # Set title from first user message
        if not conv["messages"]:
            conv["title"] = text[:40] if len(text) <= 40 else text[:37] + "…"
            self._update_sidebar()

        # User bubble
        user_b = MessageBubble(self._chat_scroll, role="user", text=text)
        user_b.grid(row=self._msg_row, column=0, sticky="ew")
        conv["messages"].append({"role": "user", "content": text})
        self._msg_row += 1

        # Bot bubble (will be filled via streaming)
        bot_b = MessageBubble(
            self._chat_scroll, role="assistant", text="",
            on_replay=self._replay_message,
        )
        bot_b.grid(row=self._msg_row, column=0, sticky="ew")
        self._msg_row += 1
        self._current_bot_bubble = bot_b
        self._scroll_bottom()

        self._status_label.configure(text="● Thinking…", text_color=SPEAKING_COL)
        self._async.submit(self._stream(text, bot_b, conv))

    async def _stream(self, query: str, bubble: MessageBubble, conv: dict) -> None:
        full = ""
        cancel = self._cancel_event

        try:
            async for token in self._brain.think(query):
                if cancel and cancel.is_set():
                    break
                full += token
                self.after(0, bubble.set_text, full)
                self.after(0, self._scroll_bottom)
        except Exception as e:
            full = f"Sorry, something went wrong: {e}"
            self.after(0, bubble.set_text, full)

        conv["messages"].append({"role": "assistant", "content": full})

        # TTS (skip if muted, interrupted, or empty)
        interrupted = cancel and cancel.is_set()
        if self._tts_mode != "off" and full and not full.startswith("Sorry,") and not interrupted:
            self.after(0, lambda: self._status_label.configure(
                text="● Speaking…", text_color=SPEAKING_COL))

            try:
                if self._tts_mode == "narrator" and self._edge_voice:
                    async def _single_yield(text):
                        yield text
                    await self._edge_voice.stream_and_speak(
                        _single_yield(full), cancel_event=cancel)
                else:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, self._speak_sync, full)
            except Exception as e:
                logger.error("TTS error: %s", e)

        self.after(0, self._finish)

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """
        Split text into sentence-sized chunks for TTS.

        Transformer TTS is O(n²) — 5 short calls is MUCH faster than 1 long one.
        Merges tiny fragments so we don't waste TTS overhead on 3-word chunks.
        """
        # Split on sentence-ending punctuation followed by whitespace
        parts = re.split(r'(?<=[.!?;:])[\s]+', text.strip())

        # Merge very short fragments (< 50 chars) with the next sentence
        MIN_CHUNK = 50
        merged: list[str] = []
        buf = ""
        for part in parts:
            part = part.strip()
            if not part:
                continue
            buf += (" " if buf else "") + part
            if len(buf) >= MIN_CHUNK:
                merged.append(buf)
                buf = ""
        if buf.strip():
            merged.append(buf.strip())

        return merged if merged else ([text.strip()] if text.strip() else [])

    def _speak_sync(self, text: str) -> None:
        """
        Sentence-by-sentence TTS with pipelined playback.

        TTS stays loaded permanently — no reload overhead.
        While sentence N plays, sentence N+1 is being synthesised.
        Stops early if _cancel_event is set.
        """
        try:
            if not self._voice.is_loaded:
                self._voice.load()

            sentences = self._split_sentences(text)
            if not sentences:
                return

            logger.info("TTS: %d sentence(s) to speak", len(sentences))
            cancel = self._cancel_event

            pending = False

            for i, sentence in enumerate(sentences):
                if cancel and cancel.is_set():
                    break

                t0 = time.perf_counter()
                audio, sr = self._voice.synthesize(sentence)
                dt = time.perf_counter() - t0
                logger.info(
                    "TTS [%d/%d] %d chars in %.1f s",
                    i + 1, len(sentences), len(sentence), dt,
                )

                if cancel and cancel.is_set():
                    break

                if pending:
                    sd.wait()

                if cancel and cancel.is_set():
                    break

                if audio.size > 0:
                    sd.play(audio, samplerate=sr)
                    pending = True
                else:
                    pending = False

            if pending and not (cancel and cancel.is_set()):
                sd.wait()

        except Exception as e:
            logger.error("TTS playback error: %s", e)

    def _finish(self) -> None:
        self._is_processing = False
        self._cancel_event = None
        # Restore send button
        self._send_btn.configure(
            text="↑", fg_color=ACCENT, hover_color=ACCENT_HOVER,
            state="normal", command=self._on_send,
        )
        self._status_label.configure(text="● Online", text_color=ACCENT)
        self._current_bot_bubble = None

    def _scroll_bottom(self) -> None:
        try:
            self._chat_scroll._parent_canvas.yview_moveto(1.0)
        except Exception:
            pass

    # ── Cleanup ─────────────────────────────────────────────────────────────

    def _on_close(self) -> None:
        logger.info("Shutting down SIRIUS.AI…")
        try:
            if self._ears and self._ears.is_loaded:
                self._ears.unload()
            if self._voice and self._voice.is_loaded:
                self._voice.unload()
        except Exception:
            pass
        self._async.stop()
        self.destroy()


# ─── Entry Point ────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(name)-12s │ %(levelname)-5s │ %(message)s",
        datefmt="%H:%M:%S",
    )

    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║              ✦  S I R I U S . A I   v 2 . 0              ║
    ║                                                            ║
    ║  Desktop Interface — ChatGPT-style dark theme             ║
    ║  Close the window or Ctrl+C to exit                       ║
    ╚════════════════════════════════════════════════════════════╝
    """)

    app = SiriusApp()
    app.mainloop()


if __name__ == "__main__":
    main()
