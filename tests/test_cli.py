from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


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


def test_router_cli_init_route_tokenize_and_benchmark(tmp_path: Path, monkeypatch, capsys) -> None:
    test_file = tmp_path / "router_test.jsonl"

    _write_jsonl(
        test_file,
        [
            {"id": "3", "text": "me ho ye", "source": "aka_asr"},
            {"id": "4", "text": "Akwaaba, yennim wo din.", "source": "pristine_twi"},
        ],
    )

    asr_path = tmp_path / "asr_tokenizer.json"
    tts_path = tmp_path / "tts_tokenizer.json"
    mixed_path = tmp_path / "mixed_tokenizer.json"
    config_path = tmp_path / "router_config.json"
    benchmark_path = tmp_path / "router_benchmark.json"

    from scripts import router as router_cli

    class DummyRouter:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def route(self, text: str):
            selected = "tts" if "," in text else "asr"
            return SimpleNamespace(
                selected_tokenizer=selected,
                domain=selected,
                confidence=0.75,
                reason="dummy",
            )

        def tokenize(self, text: str):
            return [1, 2, 3], self.route(text)

        def close(self) -> None:
            pass

    monkeypatch.setattr(router_cli, "AkanBPERouter", DummyRouter)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts/router.py",
            "init",
            "--asr-tokenizer",
            str(asr_path),
            "--tts-tokenizer",
            str(tts_path),
            "--mixed-tokenizer",
            str(mixed_path),
            "--output",
            str(config_path),
        ],
    )
    router_cli.main()
    assert "Router config saved" in capsys.readouterr().out

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts/router.py",
            "route",
            "--config",
            str(config_path),
            "--text",
            "me ho ye",
        ],
    )
    router_cli.main()
    route_output = capsys.readouterr().out
    assert "Selected:" in route_output
    assert "Domain:" in route_output

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts/router.py",
            "tokenize",
            "--config",
            str(config_path),
            "--text",
            "me ho ye",
        ],
    )
    router_cli.main()
    assert "[" in capsys.readouterr().out

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts/router.py",
            "tokenize",
            "--config",
            str(config_path),
            "--text",
            "me ho ye",
            "--show-decision",
        ],
    )
    router_cli.main()
    tokenize_with_decision_output = capsys.readouterr().out
    assert "Tokens:" in tokenize_with_decision_output
    assert "Selected:" in tokenize_with_decision_output
    assert "Confidence:" in tokenize_with_decision_output

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts/router.py",
            "benchmark",
            "--config",
            str(config_path),
            "--test-file",
            str(test_file),
            "--output",
            str(benchmark_path),
        ],
    )
    router_cli.main()
    assert "Benchmark written" in capsys.readouterr().out

    payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    assert payload["total_samples"] == 2
    assert set(payload["routing_decisions"]) == {"asr", "tts", "mixed"}
