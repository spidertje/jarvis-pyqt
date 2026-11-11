"""
Emotion plumbing — LLM emotion tags for the smoke-face frontend.

The agent injects EMOTION_PROMPT into the system prompt; the LLM starts each
reply with ``[emo:happy]`` etc. The tag is stripped before TTS/panels and
``hud.set_emotion()`` is called. If the model forgets the tag, a keyword
classifier provides a fallback so the face is never emotionless.
"""

from __future__ import annotations

import re

EMOTION_NAMES = ["neutral", "happy", "laugh", "sad", "angry", "surprised", "fear", "calm"]

EMOTION_PROMPT = (
    "\n\nEMOTION: Begin every reply with an emotion tag in brackets — one of "
    "[emo:happy] [emo:laugh] [emo:sad] [emo:angry] [emo:surprised] [emo:fear] "
    "[emo:calm] [emo:neutral] — matching the feeling of what you are saying. "
    "Put nothing before the tag. Example: '[emo:happy] Of course! Here you go.'"
)

_TAG_RE = re.compile(r"^\s*\[emo:([a-zA-Z]+)\]\s*")

_KEYWORDS = [
    ("happy", re.compile(r"\b(love|great|awesome|happy|glad|nice|wonderful|thanks|excellent|fun|yay|sure!)\b", re.I)),
    ("sad", re.compile(r"\b(sorry|sad|unfortunately|cannot|can't|fail|miss|lost|failed)\b", re.I)),
    ("angry", re.compile(r"\b(angry|furious|damn|hate|annoyed|rage|frustrat)\w*\b", re.I)),
    ("surprised", re.compile(r"\b(wow|whoa|oh my|surprise|didn't expect|unexpected|amazing)\b", re.I)),
    ("fear", re.compile(r"\b(scared|afraid|worried|danger|careful|warning|risk)\b", re.I)),
    ("calm", re.compile(r"\b(calm|relax|steady|quiet|breathe|peace|no worries)\b", re.I)),
]


def parse_emotion(text: str) -> tuple:
    """Return (emotion|None, stripped_text) for a leading [emo:x] tag."""
    m = _TAG_RE.match(text or "")
    if m:
        emo = m.group(1).lower()
        return (emo if emo in EMOTION_NAMES else None), text[m.end():]
    return None, text


def classify_emotion(text: str) -> str:
    """Keyword fallback when the LLM omits the tag."""
    for emo, rx in _KEYWORDS:
        if rx.search(text or ""):
            return emo
    return "neutral"


class EmotionTagFilter:
    """Streaming filter: strips a leading [emo:x] tag split across tokens.

    feed(token) -> cleaned text to forward (possibly empty while buffering
    the tag); flush() -> any leftover buffered text.
    """

    _PREFIX = "[emo:"

    def __init__(self):
        self._buf = ""
        self._done = False
        self.emotion = None

    def feed(self, token: str) -> str:
        if self._done:
            return token
        self._buf += token
        m = _TAG_RE.match(self._buf)
        if m:
            emo = m.group(1).lower()
            self.emotion = emo if emo in EMOTION_NAMES else classify_emotion(self._buf)
            out = self._buf[m.end():]
            self._done = True
            return out
        stripped = self._buf.lstrip()
        # abandon buffering once it can no longer be a tag prefix
        if (not self._PREFIX.startswith(stripped) and not stripped.startswith(self._PREFIX)) \
                or len(self._buf) > 16:
            self._done = True
            self.emotion = None
            return self._buf
        return ""

    def flush(self) -> str:
        if not self._done:
            self._done = True
            return self._buf
        return ""
