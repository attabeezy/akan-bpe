from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from akan_bpe.revision_manifest import normalize_text, sha256_file, validate_revision_manifest

REVISION_RUN_ID = "model__replacement__v8000__random__e1__s42"


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _dataset_record(
    path: Path,
    repo_root: Path,
    dataset_id: str,
    split: str,
    source: str,
) -> dict[str, object]:
    return {
        "id": dataset_id,
        "corpus": source,
        "split": split,
        "path": path.relative_to(repo_root).as_posix(),
        "rows": len(path.read_text(encoding="utf-8").splitlines()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "source_field": source,
    }


def _build_manifest(repo_root: Path) -> tuple[Path, dict[str, object]]:
    train_path = repo_root / "data" / "train.jsonl"
    test_path = repo_root / "data" / "test.jsonl"
    result_path = repo_root / "results" / "run.json"
    _write_jsonl(train_path, [{"id": "train-1", "text": "Maakye", "source": "fixture"}])
    _write_jsonl(test_path, [{"id": "test-1", "text": "Maadwo", "source": "fixture"}])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps({"experiment_id": REVISION_RUN_ID, "model_id": "model"}),
        encoding="utf-8",
    )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "project": {"name": "fixture"},
        "repository": {"baseline_commit": "a" * 40},
        "environments": {"local": {"status": "recorded"}},
        "naming": {
            "new_run_id_pattern": (
                r"^[a-z0-9.-]+__(original|replacement|extension)__v[0-9]+__"
                r"(random|mean_subword|preserved)__e[0-9]+(?:\.[0-9]+)?__s[0-9]+$"
            )
        },
        "datasets": [
            _dataset_record(train_path, repo_root, "train", "train", "fixture"),
            _dataset_record(test_path, repo_root, "test", "test", "fixture"),
        ],
        "protocols": {"dataset_split": {}},
        "artifacts": {
            "result": {
                "kind": "result",
                "path": result_path.relative_to(repo_root).as_posix(),
                "bytes": result_path.stat().st_size,
                "sha256": sha256_file(result_path),
            }
        },
        "runs": [
            {
                "id": REVISION_RUN_ID,
                "id_style": "revision",
                "status": "completed",
                "model_id": "model",
                "strategy": "replacement",
                "vocab_size": 8000,
                "initialization": "random",
                "epochs": 1,
                "seed": 42,
                "result_artifact": "result",
            }
        ],
        "planned_revision": {"vocabulary_ablation": {}},
        "run_log": {"completed": 1, "failed": [], "excluded": []},
        "limitations": [],
    }
    manifest_path = repo_root / "config" / "revision_manifest.yaml"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return manifest_path, manifest


def _save_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")


def test_valid_manifest_passes(tmp_path: Path) -> None:
    manifest_path, _manifest = _build_manifest(tmp_path)

    assert validate_revision_manifest(manifest_path, repo_root=tmp_path) == []


def test_artifact_tampering_fails_hash_and_size_checks(tmp_path: Path) -> None:
    manifest_path, _manifest = _build_manifest(tmp_path)
    (tmp_path / "results" / "run.json").write_text("tampered", encoding="utf-8")

    errors = validate_revision_manifest(manifest_path, repo_root=tmp_path)

    assert any("SHA-256 mismatch" in error for error in errors)
    assert any("byte-size mismatch" in error for error in errors)


def test_incorrect_dataset_row_count_fails(tmp_path: Path) -> None:
    manifest_path, manifest = _build_manifest(tmp_path)
    manifest["datasets"][0]["rows"] = 2  # type: ignore[index]
    _save_manifest(manifest_path, manifest)

    errors = validate_revision_manifest(manifest_path, repo_root=tmp_path)

    assert any("row-count mismatch" in error for error in errors)


def test_unacknowledged_train_test_text_leakage_fails(tmp_path: Path) -> None:
    manifest_path, manifest = _build_manifest(tmp_path)
    test_path = tmp_path / "data" / "test.jsonl"
    _write_jsonl(test_path, [{"id": "test-1", "text": "  MAAKYE ", "source": "fixture"}])
    manifest["datasets"][1] = _dataset_record(  # type: ignore[index]
        test_path, tmp_path, "test", "test", "fixture"
    )
    _save_manifest(manifest_path, manifest)

    errors = validate_revision_manifest(manifest_path, repo_root=tmp_path)

    assert any("dataset leakage" in error for error in errors)


def test_exact_known_leakage_is_audited_and_allowed(tmp_path: Path) -> None:
    manifest_path, manifest = _build_manifest(tmp_path)
    test_path = tmp_path / "data" / "test.jsonl"
    _write_jsonl(test_path, [{"id": "test-1", "text": "  MAAKYE ", "source": "fixture"}])
    manifest["datasets"][1] = _dataset_record(  # type: ignore[index]
        test_path, tmp_path, "test", "test", "fixture"
    )
    digest = hashlib.sha256(normalize_text("Maakye").encode("utf-8")).hexdigest()
    manifest["known_leakage"] = [
        {"normalized_text_sha256": digest, "reason": "Historical fixture overlap."}
    ]
    _save_manifest(manifest_path, manifest)

    assert validate_revision_manifest(manifest_path, repo_root=tmp_path) == []


def test_stale_known_leakage_entry_fails(tmp_path: Path) -> None:
    manifest_path, manifest = _build_manifest(tmp_path)
    manifest["known_leakage"] = [
        {"normalized_text_sha256": "b" * 64, "reason": "No longer present."}
    ]
    _save_manifest(manifest_path, manifest)

    errors = validate_revision_manifest(manifest_path, repo_root=tmp_path)

    assert any("does not match a current" in error for error in errors)


def test_invalid_revision_run_id_fails(tmp_path: Path) -> None:
    manifest_path, manifest = _build_manifest(tmp_path)
    manifest["runs"][0]["id"] = "bad-run-id"  # type: ignore[index]
    _save_manifest(manifest_path, manifest)

    errors = validate_revision_manifest(manifest_path, repo_root=tmp_path)

    assert any("does not match naming policy" in error for error in errors)


def test_duplicate_yaml_key_fails_manifest_loading(tmp_path: Path) -> None:
    manifest_path, _manifest = _build_manifest(tmp_path)
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write("\nschema_version: 1\n")

    errors = validate_revision_manifest(manifest_path, repo_root=tmp_path)

    assert any("found duplicate key 'schema_version'" in error for error in errors)


def test_repository_escape_path_fails(tmp_path: Path) -> None:
    manifest_path, manifest = _build_manifest(tmp_path)
    manifest["artifacts"]["result"]["path"] = "../outside.json"  # type: ignore[index]
    _save_manifest(manifest_path, manifest)

    errors = validate_revision_manifest(manifest_path, repo_root=tmp_path)

    assert any("escapes repository root" in error for error in errors)


def test_json_artifact_assertions_support_wildcards(tmp_path: Path) -> None:
    manifest_path, manifest = _build_manifest(tmp_path)
    result_path = tmp_path / "results" / "run.json"
    payload = {
        "experiment_id": REVISION_RUN_ID,
        "model_id": "model",
        "results": {
            "first": {"asr_test": {"num_samples": 1010}},
            "second": {"asr_test": {"num_samples": 1010}},
        },
    }
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    artifact = manifest["artifacts"]["result"]  # type: ignore[index]
    artifact["bytes"] = result_path.stat().st_size
    artifact["sha256"] = sha256_file(result_path)
    artifact["assertions"] = [
        {"path": "results.*.asr_test.num_samples", "equals": 1010}
    ]
    _save_manifest(manifest_path, manifest)

    assert validate_revision_manifest(manifest_path, repo_root=tmp_path) == []


def test_json_artifact_assertion_mismatch_fails(tmp_path: Path) -> None:
    manifest_path, manifest = _build_manifest(tmp_path)
    manifest["artifacts"]["result"]["assertions"] = [  # type: ignore[index]
        {"path": "experiment_id", "equals": "wrong"}
    ]
    _save_manifest(manifest_path, manifest)

    errors = validate_revision_manifest(manifest_path, repo_root=tmp_path)

    assert any("expected experiment_id" in error for error in errors)


def test_zero_active_leakage_exceptions_rejects_allowlist(tmp_path: Path) -> None:
    manifest_path, manifest = _build_manifest(tmp_path)
    manifest["active_protocol"] = {"zero_active_leakage_exceptions": True}
    manifest["known_leakage"] = [
        {"normalized_text_sha256": "b" * 64, "reason": "Historical overlap."}
    ]
    _save_manifest(manifest_path, manifest)

    errors = validate_revision_manifest(manifest_path, repo_root=tmp_path)

    assert any("requires zero leakage exceptions" in error for error in errors)
