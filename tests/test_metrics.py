from somax.metrics import compute_fertility


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
