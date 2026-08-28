"""Tests for ChatClient streaming (SSE) against a local mock server."""

import asyncio
import json
import socket
import threading
import time

import pytest

from jarvis.chat import ChatClient, ChatConfig


class _MockSSEServer:
    """Minimal OpenAI-compatible /chat/completions SSE server on a local port."""

    def __init__(self, reply_text="Hello there. This is a streaming reply. Final part.",
                 stream_delay=0.01, mode="sse"):
        self.reply_text = reply_text
        self.stream_delay = stream_delay
        self.mode = mode  # "sse" or "nonstream"
        self.requests: list[dict] = []
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(8)
        self.port = self._sock.getsockname()[1]
        self._running = True
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def stop(self):
        self._running = False
        try:
            self._sock.close()
        except OSError:
            pass

    def _serve(self):
        while self._running:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn):
        try:
            data = b""
            while b"\r\n\r\n" not in data:
                chunk = conn.recv(4096)
                if not chunk:
                    return
                data += chunk
            headers, _, body = data.partition(b"\r\n\r\n")
            content_length = 0
            for line in headers.decode("latin1").splitlines():
                if line.lower().startswith("content-length:"):
                    content_length = int(line.split(":", 1)[1].strip())
            while len(body) < content_length:
                body += conn.recv(4096)
            try:
                payload = json.loads(body)
            except Exception:
                payload = {}
            self.requests.append(payload)

            reply = self.reply_text
            if self.mode == "sse" and payload.get("stream"):
                # Split into word tokens (spaces preserved by the client buffer)
                tokens = reply.split()
                body_out = b""
                for i, tok in enumerate(tokens):
                    content = tok if i == 0 else " " + tok
                    event = {
                        "choices": [{"delta": {"content": content}}]
                    }
                    body_out += f"data: {json.dumps(event)}\n\n".encode()
                    time_sleep(self.stream_delay)
                body_out += b"data: [DONE]\n\n"
                resp = (
                    "HTTP/1.1 200 OK\r\n"
                    "Content-Type: text/event-stream\r\n"
                    "Cache-Control: no-cache\r\n"
                    f"Content-Length: {len(body_out)}\r\n"
                    "Connection: close\r\n\r\n"
                ).encode() + body_out
            else:
                body_out = json.dumps(
                    {
                        "choices": [
                            {"message": {"role": "assistant", "content": reply}}
                        ]
                    }
                ).encode()
                resp = (
                    "HTTP/1.1 200 OK\r\n"
                    "Content-Type: application/json\r\n"
                    f"Content-Length: {len(body_out)}\r\n"
                    "Connection: close\r\n\r\n"
                ).encode() + body_out
            conn.sendall(resp)
        except Exception:
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass


def time_sleep(seconds: float):
    time.sleep(seconds)


@pytest.fixture
def sse_server():
    server = _MockSSEServer()
    yield server
    server.stop()


@pytest.fixture
def client(sse_server):
    return ChatClient(ChatConfig(base_url=f"http://127.0.0.1:{sse_server.port}/v1"))


class TestStreamChat:
    @pytest.mark.asyncio
    async def test_stream_returns_full_text(self, client, sse_server):
        reply = await client.chat([{"role": "user", "content": "hi"}], stream=True)
        assert reply == sse_server.reply_text

    @pytest.mark.asyncio
    async def test_stream_calls_on_token_per_token(self, client, sse_server):
        tokens = []
        reply = await client.chat(
            [{"role": "user", "content": "hi"}],
            stream=True,
            on_token=lambda t: tokens.append(t),
        )
        assert reply
        assert len(tokens) >= 3
        assert "".join(tokens).strip() == sse_server.reply_text

    @pytest.mark.asyncio
    async def test_nonstream_returns_reply(self, client, sse_server):
        reply = await client.chat([{"role": "user", "content": "hi"}])
        assert reply == sse_server.reply_text

    @pytest.mark.asyncio
    async def test_stream_falls_back_when_endpoint_ignores_stream(self, client, sse_server):
        """If the endpoint returns no SSE tokens, chat() must fall back to one-shot."""
        sse_server.mode = "nonstream"
        reply = await client.chat([{"role": "user", "content": "hi"}], stream=True)
        assert reply == sse_server.reply_text


class TestStreamErrors:
    @pytest.mark.asyncio
    async def test_bad_base_url_returns_none(self):
        client = ChatClient(ChatConfig(base_url="http://127.0.0.1:1/v1", timeout=2))
        reply = await client.chat([{"role": "user", "content": "hi"}])
        assert reply is None

    @pytest.mark.asyncio
    async def test_http_error_returns_none(self):
        """A non-200 response must return None, not raise."""
        server = _MockSSEServer()
        # Replace handler to return 500
        original = server._handle

        def _500(conn):
            resp = b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}"
            conn.sendall(resp)
            conn.close()

        server._handle = _500
        client = ChatClient(ChatConfig(base_url=f"http://127.0.0.1:{server.port}/v1"))
        reply = await client.chat([{"role": "user", "content": "hi"}])
        assert reply is None
        server.stop()


class TestStreamToolCalls:
    """In-band tool_calls deltas must be surfaced via on_tool_call."""

    @pytest.fixture
    def tool_server(self):
        """SSE server that emits one tool call (split across fragments)."""
        server = _MockSSEServer()

        def _tool_stream(conn):
            data = b""
            while b"\r\n\r\n" not in data:
                chunk = conn.recv(4096)
                if not chunk:
                    return
                data += chunk
            # Fragments: id+name first, then argument JSON in two chunks.
            events = [
                {"choices": [{"delta": {"tool_calls": [
                    {"index": 0, "id": "call_abc",
                     "function": {"name": "get_weather", "arguments": '{"city": '}}]}}]},
                {"choices": [{"delta": {"tool_calls": [
                    {"index": 0,
                     "function": {"arguments": '"Rīga"'}}]}}]},
                {"choices": [{"delta": {"tool_calls": [
                    {"index": 0, "function": {"arguments": "}"}}]}}]},
                {"choices": [{"delta": {"content": "Done."}}]},
            ]
            body_out = b"".join(f"data: {json.dumps(e)}\n\n".encode() for e in events)
            body_out += b"data: [DONE]\n\n"
            resp = (
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: text/event-stream\r\n"
                f"Content-Length: {len(body_out)}\r\n"
                "Connection: close\r\n\r\n"
            ).encode() + body_out
            conn.sendall(resp)
            conn.close()

        server._handle = _tool_stream
        return server

    @pytest.mark.asyncio
    async def test_tool_call_fragments_aggregated(self, tool_server):
        client = ChatClient(ChatConfig(base_url=f"http://127.0.0.1:{tool_server.port}/v1"))
        calls = []
        reply = await client.chat(
            [{"role": "user", "content": "weather in rīga?"}],
            stream=True,
            on_tool_call=lambda k, d: calls.append((k, d)),
        )
        tool_server.stop()
        assert reply == "Done."
        # Every fragment fires the callback (accumulated snapshot each time).
        assert len(calls) == 3
        # All fragments share the stable id.
        assert all(k == "call_abc" for k, _ in calls)
        # Final snapshot has the full name and the complete argument JSON.
        last_key, last_desc = calls[-1]
        assert "get_weather" in last_desc
        assert "Rīga" in last_desc

    @pytest.mark.asyncio
    async def test_no_tool_calls_without_callback(self, tool_server):
        """No on_tool_call registered → tool deltas are ignored, text still returns."""
        client = ChatClient(ChatConfig(base_url=f"http://127.0.0.1:{tool_server.port}/v1"))
        reply = await client.chat(
            [{"role": "user", "content": "weather in rīga?"}], stream=True
        )
        tool_server.stop()
        assert reply == "Done."
