"""listen() hard cap + cancellation-safe PortAudio stream teardown.

Regression guards for the macOS segfault: the 90s voice-cycle timeout
cancelled listen() while a to_thread(stream.read) worker was still
blocked in PortAudio; the coroutine's finally then closed the stream
under that read -> use-after-free. The new structure keeps the stream
owned by one worker thread: open/read/stop/close all happen there,
cancellation cannot tear it down mid-read, and a wall-clock cap bounds
every listen.
"""

import asyncio
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from jarvis.stt import WhisperSTT, WyomingConfig


class FakeStream:
    """Blocks in read() until stop/close is called or cap hits; records
    the close-after-read ordering and lets us assert no close happened
    while a read was in flight."""

    def __init__(self, frame_rms=5000.0):
        self._stop = threading.Event()
        self._closed = threading.Event()
        self.read_in_flight = 0
        self.close_while_read = False
        self.opened_on = None
        self.closed_on = None
        self._frame_rms = frame_rms

    def start(self):
        self.opened_on = threading.get_ident()

    def read(self, frames):
        self.read_in_flight += 1
        try:
            # emulate a blocking PortAudio read
            self._stop.wait(0.25)
            if self._closed.is_set():
                self.close_while_read = True  # close landed mid-read = UAF
                raise RuntimeError("stream closed mid-read")
            data = (np.random.default_rng(1).integers(
                -int(self._frame_rms), int(self._frame_rms), frames)
                .astype(np.int16))
            return data, False
        finally:
            self.read_in_flight -= 1

    def stop(self):
        pass

    def close(self):
        self.closed_on = threading.get_ident()
        assert self.read_in_flight == 0, "close() while a read is in flight"
        self._closed.set()
        self._stop.set()


def _fake_sd(stream):
    return SimpleNamespace(InputStream=lambda **kw: stream)


@pytest.mark.asyncio
async def test_listen_hard_cap_stops_noisy_room():
    """Noise never falls silent -> the wall-clock cap still ends the listen."""
    stt = WhisperSTT(WyomingConfig())
    stt._connected = True
    stt._writer = SimpleNamespace(is_closing=lambda: False)
    stream = FakeStream(frame_rms=5000)  # always 'speech', never silent
    stt._get_sd = lambda: _fake_sd(stream)
    stt.transcribe = _async_ret("hi")

    t0 = time.monotonic()
    out = await stt.listen(timeout=0.6, silence_threshold=100.0)  # cap 0.6s
    dt = time.monotonic() - t0
    assert out == "hi"
    assert dt < 3.0, f"listen ran {dt:.1f}s despite a 0.6s cap"


def _async_ret(val):
    async def _f(*a, **k):
        return val
    return _f


@pytest.mark.asyncio
async def test_should_stop_releases_mic():
    stt = WhisperSTT(WyomingConfig())
    stt._connected = True
    stt._writer = SimpleNamespace(is_closing=lambda: False)
    stream = FakeStream(frame_rms=5000)
    stt._get_sd = lambda: _fake_sd(stream)
    stt.transcribe = _async_ret("x")

    flag = {"stop": False}

    async def stopper():
        await asyncio.sleep(0.3)
        flag["stop"] = True

    t = asyncio.create_task(stopper())
    out = await stt.listen(
        timeout=30.0, silence_threshold=100.0,
        should_stop=lambda: flag["stop"])
    await t
    assert out == "x"
    assert stream.closed_on is not None


@pytest.mark.asyncio
async def test_cancel_does_not_close_stream_under_read():
    """Simulated 90s-cycle cancel: worker keeps the stream, closes only
    after its last read returns — the macOS use-after-free guard."""
    stt = WhisperSTT(WyomingConfig())
    stt._connected = True
    stt._writer = SimpleNamespace(is_closing=lambda: False)
    stream = FakeStream(frame_rms=5000)
    stt._get_sd = lambda: _fake_sd(stream)
    stt.transcribe = _async_ret("x")

    task = asyncio.create_task(
        stt.listen(timeout=30.0, silence_threshold=100.0))
    await asyncio.sleep(0.7)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # The shielded worker must still have released the stream cleanly,
    # on the same thread that opened it, never mid-read.
    deadline = time.monotonic() + 2.0
    while stream.closed_on is None and time.monotonic() < deadline:
        await asyncio.sleep(0.05)
    assert stream.closed_on is not None, "stream never closed after cancel"
    assert stream.closed_on == stream.opened_on
    assert not stream.close_while_read
