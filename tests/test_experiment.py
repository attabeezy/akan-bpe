import json
from pathlib import Path

from somax.experiment import ExperimentTokenizer, run_fertility_experiment
from somax.tokenizers import train_bpe_tokenizer


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_run_fertility_experiment_returns_unified_payload(tmp_path: Path) -> None:
    asr_file = tmp_path / "aka_asr_test.jsonl"
    tts_file = tmp_path / "pristine_twi_test.jsonl"
    _write_jsonl(asr_file, [{"id": "1", "text": "uhm chale", "source": "aka_asr"}])
    _write_jsonl(tts_file, [{"id": "2", "text": "akwaaba ma me", "source": "pristine_twi"}])

    control_path = tmp_path / "control_tokenizer.json"
    asr_path = tmp_path / "asr_tokenizer.json"
    tts_path = tmp_path / "tts_tokenizer.json"
    for path, texts, name in [
        (control_path, ["uhm chale", "akwaaba ma me"], "control"),
        (asr_path, ["uhm chale"], "asr"),
        (tts_path, ["akwaaba ma me"], "tts"),
    ]:
        train_bpe_tokenizer(texts=texts, output_path=path, vocab_size=64, name=name)

    payload = run_fertility_experiment(
        experiment_id="exp001",
        tokenizers=[
            ExperimentTokenizer(name="control", reference=str(control_path)),
            ExperimentTokenizer(name="asr", reference=str(asr_path)),
            ExperimentTokenizer(name="tts", reference=str(tts_path)),
        ],
        asr_test_file=str(asr_file),
        tts_test_file=str(tts_file),
    )

    assert payload["experiment_id"] == "exp001"
    assert set(payload["results"].keys()) == {"control", "asr", "tts"}
    assert "asr_test" in payload["results"]["control"]
    assert "tts_test" in payload["results"]["control"]
