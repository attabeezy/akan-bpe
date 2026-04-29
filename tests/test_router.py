"""Tests for Akan-BPE router."""

import pytest

from akan_bpe.router import AkanBPERouter


@pytest.fixture
def router():
    return AkanBPERouter(
        asr_tokenizer_path="models/asr_tokenizer.json",
        tts_tokenizer_path="models/tts_tokenizer.json",
        mixed_tokenizer_path="models/mixed_tokenizer.json",
    )


class TestDetectDomain:
    def test_detect_formal_text_with_punctuation(self, router):
        text = "Wɔn a wɔgyina agoprama no do no hyɛ ataare fufuo."
        domain, confidence = router.detect_domain(text)
        assert domain in ("asr", "tts", "mixed")

    def test_detect_conversational_no_punctuation(self, router):
        text = "me ho da me hɔ"
        domain, confidence = router.detect_domain(text)
        assert domain in ("asr", "tts", "mixed")

    def test_detect_short_words_asr(self, router):
        text = "yea oh yes"
        domain, confidence = router.detect_domain(text)
        assert domain in ("asr", "tts", "mixed")

    def test_detect_long_words_tts(self, router):
        text = "this is a very long sentence with many complex words"
        domain, confidence = router.detect_domain(text)
        assert domain in ("asr", "tts", "mixed")

    def test_empty_text_returns_mixed(self, router):
        text = ""
        domain, confidence = router.detect_domain(text)
        assert domain == "mixed"

    def test_unicode_heuristic(self, router):
        text = "mebata"
        domain, confidence = router.detect_domain(text)
        assert domain in ("asr", "tts", "mixed")


class TestRoute:
    def test_route_returns_routing_decision(self, router):
        text = "Wɔn a wɔgyina agoprama no do no hyɛ ataare fufuo."
        decision = router.route(text)
        assert decision.selected_tokenizer in ("asr", "tts", "mixed")
        assert decision.confidence > 0
        assert decision.domain in ("asr", "tts", "mixed")


class TestTokenize:
    def test_tokenize_returns_tokens(self, router):
        text = "me ho da me hɔ"
        tokens, decision = router.tokenize(text)
        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_tokenize_with_mux(self, router):
        text = "me ho da me hɔ"
        tokens, info = router.tokenize_with_mux(text)
        assert isinstance(tokens, list)
        assert len(tokens) > 0
