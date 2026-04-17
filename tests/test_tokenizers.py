from pathlib import Path

from somax.tokenizers import build_tokenizer_stats, load_tokenizer, train_bpe_tokenizer


def test_train_bpe_tokenizer_writes_output(tmp_path: Path) -> None:
    output_path = tmp_path / "asr_tokenizer.json"
    info = train_bpe_tokenizer(
        texts=["hello world", "akwaaba"],
        output_path=output_path,
        vocab_size=64,
        name="asr",
    )

    assert output_path.exists()
    assert info.name == "asr"
    assert info.output_path == str(output_path)


def test_build_tokenizer_stats_contains_histogram(tmp_path: Path) -> None:
    output_path = tmp_path / "tts_tokenizer.json"
    info = train_bpe_tokenizer(
        texts=["one two", "three four five"],
        output_path=output_path,
        vocab_size=64,
        name="tts",
    )

    stats = build_tokenizer_stats(info, ["one two", "three four five"])
    assert "word_count_histogram" in stats
    assert stats["num_texts"] == 2


def test_load_tokenizer_reads_local_json(tmp_path: Path) -> None:
    output_path = tmp_path / "mixed_tokenizer.json"
    train_bpe_tokenizer(
        texts=["hello world", "this is a test"],
        output_path=output_path,
        vocab_size=64,
        name="mixed",
    )

    tokenizer = load_tokenizer(str(output_path))
    encoded = tokenizer.encode("hello world")
    assert isinstance(encoded, list)
    assert encoded
