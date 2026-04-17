from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_train_bpe_cli_and_benchmark_cli(tmp_path: Path) -> None:
    asr_train = tmp_path / "aka_asr_train.jsonl"
    tts_train = tmp_path / "pristine_twi_train.jsonl"
    asr_test = tmp_path / "aka_asr_test.jsonl"
    tts_test = tmp_path / "pristine_twi_test.jsonl"

    _write_jsonl(asr_train, [{"id": "1", "text": "uhm chale", "source": "aka_asr"}])
    _write_jsonl(tts_train, [{"id": "2", "text": "akwaaba ma me", "source": "pristine_twi"}])
    _write_jsonl(asr_test, [{"id": "3", "text": "uhm chale", "source": "aka_asr"}])
    _write_jsonl(tts_test, [{"id": "4", "text": "akwaaba ma me", "source": "pristine_twi"}])

    control_path = tmp_path / "control_tokenizer.json"
    asr_path = tmp_path / "asr_tokenizer.json"
    tts_path = tmp_path / "tts_tokenizer.json"
    output_path = tmp_path / "experiment.json"
    repo_root = Path(__file__).resolve().parents[1]

    for name, input_path, tokenizer_path in [
        ("control", asr_train, control_path),
        ("asr", asr_train, asr_path),
        ("tts", tts_train, tts_path),
    ]:
        subprocess.run(
            [
                sys.executable,
                "scripts/train_bpe.py",
                "--inputs",
                str(input_path),
                "--output",
                str(tokenizer_path),
                "--name",
                name,
                "--vocab-size",
                "64",
            ],
            check=True,
            cwd=repo_root,
        )

    subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_fertility.py",
            "--experiment-id",
            "exp_cli",
            "--control-tokenizer",
            str(control_path),
            "--asr-tokenizer",
            str(asr_path),
            "--tts-tokenizer",
            str(tts_path),
            "--asr-test-file",
            str(asr_test),
            "--tts-test-file",
            str(tts_test),
            "--output",
            str(output_path),
        ],
        check=True,
        cwd=repo_root,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["experiment_id"] == "exp_cli"
    assert "summary" in payload
