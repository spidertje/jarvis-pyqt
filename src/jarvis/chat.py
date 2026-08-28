"""
Jarvis Chat — OpenAI-compatible API client.

Talks to any OpenAI-format LLM endpoint (local Ollama, LM Studio, Hindsight, etc.)
"""

import json
import logging
from dataclasses import dataclass

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class ChatConfig:
    """LLM endpoint configuration."""

    base_url: str | None = None
    api_key: str | None = None
    model: str = "auto"
    temperature: float = 0.7
    max_tokens: int = 1024
    system_prompt: str = "You are Jarvis, a helpful AI assistant."
    timeout: int = 120  # seconds


class ChatClient:
    """Simple chat client for OpenAI-compatible APIs."""

    def __init__(self, config: ChatConfig | None = None):
        self.config = config or ChatConfig()
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            headers = {}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            timeout = (
                aiohttp.ClientTimeout(total=self.config.timeout) if self.config.timeout else None
            )
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
        messages: list[dict[str, str]],
        stream: bool = False,
        system_prompt: str | None = None,
        on_token=None,
        on_tool_call=None,
    ) -> str | None:
        """
        Send chat messages and get a response.

        Args:
            messages: List of {"role": "user"|"assistant", "content": "..."}
            stream: If True, tokens are delivered via ``on_token`` as they arrive.
            system_prompt: Optional system prompt to prepend.
            on_token: Optional callback(text_chunk) called for each streamed token.
            on_tool_call: Optional callback(dict) called for each tool call the
                LLM emits in-band (OpenAI ``tool_calls`` deltas, aggregated by id).

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
                result = await self._stream_chat(
                    session, payload, on_token=on_token, on_tool_call=on_tool_call
                )
                if result is not None:
                    return result
                # Endpoint ignored `stream` (or sent nothing) — fall back to one-shot
                logger.info("Stream returned nothing — falling back to non-streaming")
                return await self._nonstream_chat(session, {**payload, "stream": False})
            return await self._nonstream_chat(session, payload)
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return None

    async def _nonstream_chat(self, session: aiohttp.ClientSession, payload: dict) -> str | None:
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
        on_tool_call=None,
    ) -> str | None:
        """
        Stream tokens from the API (SSE ``data:`` chunks).

        Args:
            on_token: Optional callback(text_chunk) for each token.
            on_tool_call: Optional callback(key, description) for tool calls the
                LLM emits in-band. ``key`` is the stable tool-call id;
                ``description`` is the running "name(args)" snapshot, which is
                re-sent (accumulated) on each fragment of a streaming call.

        Returns:
            Full response text, or None if no tokens were received.
        """
        if not self.config.base_url:
            logger.error("No LLM base_url configured")
            return None
        payload = {**payload, "stream": True}
        chunks: list[str] = []
        tool_calls: dict[str, dict] = {}  # index -> {id, name, arguments}
        url = f"{self.config.base_url}/chat/completions"
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                body = await resp.text()
                logger.error(f"Stream API returned {resp.status}: {body}")
                return None
            async for line in resp.content:
                line = line.strip()
                if not line:
                    continue
                if isinstance(line, bytes):
                    line = line.decode("utf-8", "replace")
                if not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
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
                    # In-band tool calls (OpenAI tool_calls deltas).
                    for tc in delta.get("tool_calls", []) or []:
                        info = tool_calls.setdefault(tc.get("index", 0), {
                            "id": "", "name": "", "arguments": ""
                        })
                        frag = tc.get("function") or {}
                        if tc.get("id"):
                            info["id"] = tc["id"]
                        if frag.get("name"):
                            info["name"] = frag["name"]
                        if frag.get("arguments"):
                            info["arguments"] += frag["arguments"]
                        if on_tool_call:
                            on_tool_call(
                                info["id"] or f"tc-{tc.get('index', 0)}",
                                self._describe_tool_call(info),
                            )
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
        return "".join(chunks) if chunks else None

    @staticmethod
    def _describe_tool_call(info: dict) -> str:
        """Human-readable one-line description of a (partial) tool call."""
        name = info.get("name") or "…"
        args = (info.get("arguments") or "").strip()
        # Trim long argument JSON so the HUD line stays readable.
        if len(args) > 48:
            args = args[:47] + "…"
        return f"▸ {name}({args})" if args else f"▸ {name}(…)"
