"""Tests for streaming text utilities (sentence splitting for progressive TTS).

Semantics under test: a sentence is emitted the moment it is unambiguously
complete (a non-digit, non-abbreviation character follows the terminal mark).
A mark at the very end of the buffer is HELD; the caller releases the tail
with flush() once the stream ends.
"""

from jarvis.streaming import SentenceBuffer, split_sentences


class TestSplitSentences:
    def test_single_sentence(self):
        assert split_sentences("Hello world.") == ["Hello world."]

    def test_multiple_sentences(self):
        assert split_sentences("First. Second! Third?") == [
            "First.",
            "Second!",
            "Third?",
        ]

    def test_no_punctuation(self):
        assert split_sentences("no punctuation here") == ["no punctuation here"]

    def test_empty(self):
        assert split_sentences("") == []

    def test_decimal_not_split(self):
        result = split_sentences("The answer is 3.14. Done.")
        assert "The answer is 3.14." in result
        assert "Done." in result

    def test_trailing_sentence_returned(self):
        # Whole-text mode releases the tail.
        assert split_sentences("One. Two.") == ["One.", "Two."]

    def test_abbreviation_not_split(self):
        assert split_sentences("Use e.g. apple juice. Done.") == [
            "Use e.g. apple juice.",
            "Done.",
        ]


class TestSentenceBuffer:
    def test_sentences_emitted_when_next_token_arrives(self):
        buf = SentenceBuffer()
        out = []
        for tok in ["Hello", " world", ".", " Then", " next", "."]:
            out.extend(buf.feed(tok))
        # "Hello world." released by the space after it; the trailing
        # "Then next." is held until flush.
        assert out == ["Hello world."]
        assert buf.flush() == "Then next."

    def test_streaming_char_by_char(self):
        buf = SentenceBuffer()
        text = "Short reply. A longer second sentence follows here."
        out = []
        for ch in text:
            out.extend(buf.feed(ch))
        assert out == ["Short reply."]
        assert buf.flush() == "A longer second sentence follows here."

    def test_flush_returns_remainder(self):
        buf = SentenceBuffer()
        assert buf.flush() == ""
        buf.feed("incomplete")
        assert buf.flush() == "incomplete"
        assert buf.flush() == ""  # cleared

    def test_decimal_held_then_released(self):
        """'3.' must not be emitted while digits may still follow."""
        buf = SentenceBuffer()
        assert buf.feed("pi is 3.") == []  # held
        out = buf.feed("14159. end")
        assert out == ["pi is 3.14159."]
        assert buf.flush() == "end"

    def test_decimal_boundary_at_space(self):
        """'3.' + '5' is a decimal (held); '3.' + ' done' is a real boundary."""
        buf = SentenceBuffer()
        assert buf.feed("Answer: 3.") == []   # held — digit may follow
        assert buf.feed("5") == []            # still a decimal, held at buffer end
        assert buf.flush() == "Answer: 3.5"

        buf2 = SentenceBuffer()
        assert buf2.feed("Answer: 3.") == []
        assert buf2.feed(" done") == ["Answer: 3."]

    def test_abbreviation_held(self):
        """The dots inside 'e.g.' must not split the sentence."""
        buf = SentenceBuffer()
        assert buf.feed("Try e.g.") == []  # both dots held
        assert buf.feed(" apple juice") == []  # still one sentence
        assert buf.feed(" and more") == []  # no boundary yet
        assert buf.flush() == "Try e.g. apple juice and more"

    def test_single_word_boundary_still_splits(self):
        """'ok. hi' — a dot between a word and a space+letter still splits."""
        buf = SentenceBuffer()
        out = buf.feed("ok.")
        assert out == []  # held at buffer end
        assert buf.feed(" hi") == ["ok."]

    def test_realistic_token_stream(self):
        """Simulated LLM token stream."""
        buf = SentenceBuffer()
        tokens = [
            "Sure!", " Here's", " the", " plan:",
            " step", " one", ".", " step", " two", ".",
        ]
        out = []
        for t in tokens:
            out.extend(buf.feed(t))
        assert out == ["Sure!", "Here's the plan: step one."]
        assert buf.flush() == "step two."

    def test_multiple_sentences_one_stream(self):
        buf = SentenceBuffer()
        text = "First sentence. Second sentence! Third sentence?"
        out = []
        for ch in text:
            out.extend(buf.feed(ch))
        assert out == ["First sentence.", "Second sentence!"]
        assert buf.flush() == "Third sentence?"
