import math

from akan_bpe.metrics import bits_per_byte, compute_fertility, count_utf8_bytes


class DummyTokenizer:
    def encode(self, text: str) -> list[str]:
        return text.split()


def test_compute_fertility_uses_tokens_per_word() -> None:
    result = compute_fertility(
        tokenizer_name="dummy",
        tokenizer_ref="dummy",
        test_set_name="test",
        source_file="fixture.jsonl",
        texts=["one two", "three four five"],
        tokenizer=DummyTokenizer(),
    )

    assert result.total_tokens == 5
    assert result.total_words == 5
    assert result.fertility == 1.0


def test_count_utf8_bytes_counts_multibyte_akan_chars() -> None:
    # ASCII "abc" = 3 bytes; "ɛɔ" are 2-byte UTF-8 chars = 4 bytes.
    assert count_utf8_bytes(["abc"]) == 3
    assert count_utf8_bytes(["ɛɔ"]) == 4
    assert count_utf8_bytes(["abc", "ɛɔ"]) == 7
    assert count_utf8_bytes([]) == 0


def test_bits_per_byte_converts_nats_to_bits_per_byte() -> None:
    # 1 token costing ln(2) nats == 1 bit; over 2 bytes that is 0.5 bpb.
    assert bits_per_byte(math.log(2), 2) == 0.5
    # 4 bits over 8 bytes.
    assert bits_per_byte(4 * math.log(2), 8) == 0.5


def test_bits_per_byte_guards_zero_bytes() -> None:
    assert bits_per_byte(10.0, 0) == 0.0
