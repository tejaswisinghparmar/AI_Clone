"""
brain.py — LLM Client via Ollama
=================================
Async streaming wrapper around a custom Ollama model
built from gemma3:12b (Q4_K_M quantisation).

The system prompt / personality is baked into the Modelfile,
so this module only handles conversation history and streaming.
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator

from ollama import AsyncClient

logger = logging.getLogger(__name__)

# ─── Configuration ──────────────────────────────────────────────────────────────
OLLAMA_MODEL    = "my-ai-clone"            # name given in `ollama create`
OLLAMA_HOST     = "http://localhost:11434"
MAX_HISTORY     = 14                       # turns — smaller = faster context


class Brain:
    """
    Async Ollama LLM client with sliding-window conversation memory.

    Usage:
        brain = Brain()
        async for token in brain.think("What's up?"):
            print(token, end="")
    """

    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        host: str = OLLAMA_HOST,
        max_history: int = MAX_HISTORY,
    ):
        self.model = model
        self.host = host
        self.max_history = max_history
        self._client = AsyncClient(host=host)
        self._history: list[dict[str, str]] = []
        self._max_messages = max_history * 2   # pre-compute once

    # ── History management ──────────────────────────────────────────────────────

    def clear_history(self) -> None:
        """Wipe conversation memory."""
        self._history.clear()
        logger.info("Conversation history cleared")

    def _trim_history(self) -> None:
        """Keep only the last *max_history* turns to stay within context."""
        if len(self._history) > self._max_messages:
            self._history = self._history[-self._max_messages:]

    # ── Streaming chat ──────────────────────────────────────────────────────────

    async def think(self, user_input: str) -> AsyncGenerator[str, None]:
        """
        Send *user_input* to Ollama and **yield** text tokens as they arrive.

        The full response is appended to conversation history once the
        stream is exhausted.
        """
        self._history.append({"role": "user", "content": user_input})
        self._trim_history()

        full_response = ""

        try:
            stream = await self._client.chat(
                model=self.model,
                messages=self._history,
                stream=True,
                keep_alive="10m",      # keep model warm between turns
            )

            async for chunk in stream:
                token = chunk["message"]["content"]
                full_response += token
                yield token

        except Exception as exc:
            logger.error("Ollama error: %s", exc)
            fallback = "Hmm, my brain just glitched. Say that again?"
            full_response = fallback
            yield fallback

        # Persist the assistant turn
        self._history.append({"role": "assistant", "content": full_response})

    # ── Non-streaming convenience ───────────────────────────────────────────────

    async def think_all(self, user_input: str) -> str:
        """Return the complete response as a single string."""
        parts: list[str] = []
        async for token in self.think(user_input):
            parts.append(token)
        return "".join(parts)

    # ── Health check ────────────────────────────────────────────────────────────

    async def is_alive(self) -> bool:
        """
        Verify that Ollama is reachable and our custom model is available.
        """
        try:
            response = await self._client.list()
            available = [m.model for m in response.models]

            # Check for exact match or partial match (ollama may append :latest)
            found = any(
                self.model == name or self.model in name
                for name in available
            )

            if not found:
                logger.warning(
                    "Model '%s' not found.  Available: %s\n"
                    "  → Run:  ollama create %s -f Modelfile",
                    self.model, available, self.model,
                )
                return False

            logger.info("Ollama model '%s' is available", self.model)
            return True

        except Exception as exc:
            logger.error("Cannot reach Ollama at %s: %s", self.host, exc)
            return False
