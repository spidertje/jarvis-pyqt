"""Streaming text utilities — sentence splitting for progressive TTS.

The LLM reply is consumed token-by-token; :class:`SentenceBuffer` accumulates
tokens and yields complete sentences as soon as they are ready so the TTS can
start speaking the first sentence while the rest of the reply is still
streaming.

A terminal mark (``.!?…``) is a **sentence boundary** when all hold:

* a character follows it in the buffer (a mark at the very end of the
  buffer is **held** — the stream may still deliver more, or the producer
  releases the tail with :meth:`SentenceBuffer.flush`),
* the following character is **not a digit** (decimal: ``3.`` + ``14``), and
* the mark is **not** part of a single-letter dot-abbreviation — the first
  dot of ``x.y`` (``e.g.``) or the second dot of ``x.y.`` (``i.e.``).

So ``3.14`` and ``e.g. apple`` stay whole, while ``3. Done`` and normal
prose (``…step one. step two``) split correctly.
"""

import re

_SENT_END = re.compile(r"[.!?…]+")


def _single(c: str | None) -> bool:
    """True if ``c`` is a single lowercase letter."""
    return c is not None and "a" <= c <= "z"


def _is_abbreviation_dot(buf: str, p: int, nxt: str | None) -> bool:
    """True if the mark at index ``p`` is part of ``x.y`` / ``x.y.``."""
    prev1 = buf[p - 1] if p >= 1 else None
    prev2 = buf[p - 2] if p >= 2 else None
    if _single(prev1) and _single(nxt):
        return True  # first dot of x.y   (e.g.)
    if prev2 == "." and _single(prev1):
        return True  # second dot of x.y. (i.e.)
    return False


def _end_index(buf: str) -> int | None:
    """Index just past the first *safe* sentence end in ``buf``, or None."""
    for m in _SENT_END.finditer(buf):
        p, end = m.start(), m.end()
        nxt = buf[end] if end < len(buf) else None
        if nxt is None:
            # End of buffer: the stream may still deliver digits (decimals)
            # or more text — hold; the producer releases the tail with flush().
            continue
        if nxt.isdigit():
            # Decimal point: 3.14
            continue
        if _is_abbreviation_dot(buf, p, nxt):
            # Dot-abbreviation: e.g. / i.e. / o.k.
            continue
        return end
    return None


def split_sentences(text: str) -> list[str]:
    """Split text into sentences (best effort, decimal/abbreviation-safe).

    Whole-text mode: the trailing sentence is always returned (released via
    flush), so this is safe for non-streaming replies too.
    """
    if not text:
        return []
    buf = SentenceBuffer()
    out = list(buf.feed(text))
    tail = buf.flush()
    if tail:
        out.append(tail)
    return out


class SentenceBuffer:
    """Accumulate streamed tokens and emit complete sentences.

    A sentence is emitted as soon as it is unambiguously complete. The
    trailing mark of the current buffer is held — the producer releases it
    with :meth:`flush` once the LLM stream ends (or when the reply is a
    one-shot non-streamed text).

    Usage::

        buf = SentenceBuffer()
        for token in stream:
            for sentence in buf.feed(token):
                speak(sentence)
        # stream ended
        tail = buf.flush()
        if tail:
            speak(tail)
    """

    def __init__(self) -> None:
        self._buf = ""

    def feed(self, token: str) -> list[str]:
        """Append a token; return any sentences that became complete."""
        self._buf += token
        out: list[str] = []
        while True:
            end = _end_index(self._buf)
            if end is None:
                break
            sentence = self._buf[:end].strip()
            self._buf = self._buf[end:].lstrip()
            if sentence:
                out.append(sentence)
        return out

    def flush(self) -> str:
        """Return and clear any remaining incomplete text."""
        text = self._buf.strip()
        self._buf = ""
        return text
