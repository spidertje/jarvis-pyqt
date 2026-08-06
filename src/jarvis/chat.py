"""
Jarvis Chat — OpenAI-compatible API client.

Talks to any OpenAI-format LLM endpoint (local Ollama, LM Studio, Hindsight, etc.)
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class ChatConfig:
    """LLM endpoint configuration."""
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model: str = "auto"
    temperature: float = 0.7
    max_tokens: int = 1024
    system_prompt: str = "You are Jarvis, a helpful AI assistant."
    timeout: int = 120  # seconds


class ChatClient:
    """Simple chat client for OpenAI-compatible APIs."""

    def __init__(self, config: Optional[ChatConfig] = None):
        self.config = config or ChatConfig()
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            headers = {}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            timeout = aiohttp.ClientTimeout(total=self.config.timeout) if self.config.timeout else None
            self._session = aiohttp.ClientSession(
                headers=headers,
                timeout=timeout,
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def chat(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        system_prompt: Optional[str] = None,
    ) -> Optional[str]:
        """
        Send chat messages and get a response.

        Args:
            messages: List of {"role": "user"|"assistant", "content": "..."}
            stream: If True, returns streamed token chunks via callback.
            system_prompt: Optional system prompt to prepend.

        Returns:
            Complete text response, or None on failure.
        """
        session = await self._get_session()

        # Build messages list with optional system prompt
        chat_messages = list(messages)
        if system_prompt:
            chat_messages.insert(0, {"role": "system", "content": system_prompt})

        payload = {
            "model": self.config.model,
            "messages": chat_messages,
            "temperature": self.config.temperature,
            "stream": stream,
        }

        try:
            if stream:
                return await self._stream_chat(session, payload)
            else:
                return await self._nonstream_chat(session, payload)
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return None

    async def _nonstream_chat(self, session: aiohttp.ClientSession, payload: dict) -> Optional[str]:
        """Get a complete response in one shot."""
        if not self.config.base_url:
            logger.error("No LLM base_url configured")
            return None
        url = f"{self.config.base_url}/chat/completions"
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                body = await resp.text()
                logger.error(f"Chat API returned {resp.status}: {body}")
                return None
            data = await resp.json()
            return data["choices"][0]["message"]["content"]

    async def _stream_chat(
        self,
        session: aiohttp.ClientSession,
        payload: dict,
        on_token=None,
    ) -> Optional[str]:
        """
        Stream tokens from the API.

        Args:
            on_token: Optional callback(text_chunk) for each token.
        """
        if not self.config.base_url:
            logger.error("No LLM base_url configured")
            return None
        payload["stream"] = True
        chunks = []
        url = f"{self.config.base_url}/chat/completions"
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                body = await resp.text()
                logger.error(f"Stream API returned {resp.status}: {body}")
                return None
            async for line in resp.content:
                line = line.strip()
                if not line or not line.startswith(b"data: "):
                    continue
                data_str = line[6:].decode("utf-8")
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk["choices"][0].get("delta", {})
                    token = delta.get("content", "")
                    if token:
                        chunks.append(token)
                        if on_token:
                            on_token(token)
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
        return "".join(chunks)
