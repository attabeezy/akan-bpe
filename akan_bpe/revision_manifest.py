"""Validation helpers for the frozen revision experiment manifest."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import yaml
from yaml.constructor import ConstructorError

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
REQUIRED_SECTIONS = {
    "schema_version",
    "project",
    "repository",
    "environments",
    "naming",
    "datasets",
    "protocols",
    "artifacts",
    "runs",
    "run_log",
    "planned_revision",
    "limitations",
}


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: yaml.SafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def load_revision_manifest(path: Path) -> dict[str, Any]:
    """Load a revision manifest with YAML's safe loader."""
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.load(handle, Loader=_UniqueKeySafeLoader)
    if not isinstance(payload, dict):
        raise ValueError("Manifest root must be a mapping.")
    return payload


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file's raw bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    """Normalize text for conservative cross-split leakage checks."""
    normalized = unicodedata.normalize("NFC", text)
    return " ".join(normalized.split()).casefold()


def _safe_repo_path(repo_root: Path, relative_path: object) -> tuple[Path | None, str | None]:
    if not isinstance(relative_path, str) or not relative_path:
        return None, "path must be a non-empty string"
    candidate = Path(relative_path)
    if candidate.is_absolute():
        return None, f"path must be repository-relative: {relative_path}"
    resolved_root = repo_root.resolve()
    resolved = (repo_root / candidate).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        return None, f"path escapes repository root: {relative_path}"
    return resolved, None


def _validate_digest(
    *,
    label: str,
    record: Mapping[str, Any],
    path: Path,
    errors: list[str],
) -> None:
    expected = record.get("sha256")
    if not isinstance(expected, str) or not SHA256_RE.fullmatch(expected):
        errors.append(f"{label}: sha256 must be 64 lowercase hexadecimal characters")
        return
    actual = sha256_file(path)
    if actual != expected:
        errors.append(f"{label}: SHA-256 mismatch (expected {expected}, got {actual})")


def _iter_experiment_records(value: object) -> Iterator[Mapping[str, Any]]:
    if isinstance(value, dict):
        experiment_id = value.get("experiment_id")
        if isinstance(experiment_id, str):
            yield value
        for nested in value.values():
            yield from _iter_experiment_records(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _iter_experiment_records(nested)


def _is_not_recorded(value: object) -> bool:
    return (
        isinstance(value, dict)
        and value.get("status") == "not_recorded"
        and isinstance(value.get("note"), str)
        and bool(value["note"].strip())
    )


def _json_path_values(value: object, parts: list[str]) -> Iterator[object]:
    if not parts:
        yield value
        return
    head, *tail = parts
    if head == "*":
        if isinstance(value, dict):
            for nested in value.values():
                yield from _json_path_values(nested, tail)
        elif isinstance(value, list):
            for nested in value:
                yield from _json_path_values(nested, tail)
    elif isinstance(value, dict) and head in value:
        yield from _json_path_values(value[head], tail)


def _validate_json_assertions(
    *,
    label: str,
    payload: object,
    assertions: object,
    errors: list[str],
) -> None:
    if assertions is None:
        return
    if not isinstance(assertions, list):
        errors.append(f"{label}: assertions must be a list")
        return
    for index, assertion in enumerate(assertions):
        assertion_label = f"{label}.assertions[{index}]"
        if not isinstance(assertion, dict):
            errors.append(f"{assertion_label}: assertion must be a mapping")
            continue
        json_path = assertion.get("path")
        if not isinstance(json_path, str) or not json_path:
            errors.append(f"{assertion_label}: path must be a non-empty string")
            continue
        if "equals" not in assertion:
            errors.append(f"{assertion_label}: equals is required")
            continue
        values = list(_json_path_values(payload, json_path.split(".")))
        if not values:
            errors.append(f"{assertion_label}: JSON path matched no values: {json_path}")
            continue
        expected = assertion["equals"]
        for value in values:
            if value != expected:
                errors.append(
                    f"{assertion_label}: expected {json_path} == {expected!r}, got {value!r}"
                )


def _validate_dataset_records(
    records: object,
    known_leakage: object,
    repo_root: Path,
    errors: list[str],
) -> None:
    if not isinstance(records, list) or not records:
        errors.append("datasets must be a non-empty list")
        return

    seen_dataset_ids: set[str] = set()
    global_ids: dict[str, str] = {}
    train_texts: dict[str, str] = {}
    evaluation_texts: dict[str, str] = {}
    allowed_text_digests: dict[str, str] = {}
    found_allowed_digests: set[str] = set()

    if known_leakage is not None:
        if not isinstance(known_leakage, list):
            errors.append("known_leakage must be a list")
        else:
            for index, item in enumerate(known_leakage):
                label = f"known_leakage[{index}]"
                if not isinstance(item, dict):
                    errors.append(f"{label}: record must be a mapping")
                    continue
                digest = item.get("normalized_text_sha256")
                reason = item.get("reason")
                if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
                    errors.append(f"{label}: normalized_text_sha256 must be a SHA-256 digest")
                elif not isinstance(reason, str) or not reason.strip():
                    errors.append(f"{label}: reason must be a non-empty string")
                else:
                    allowed_text_digests[digest] = reason

    for index, raw_record in enumerate(records):
        label = f"datasets[{index}]"
        if not isinstance(raw_record, dict):
            errors.append(f"{label}: record must be a mapping")
            continue
        record = raw_record
        dataset_id = record.get("id")
        split = record.get("split")
        if not isinstance(dataset_id, str) or not dataset_id:
            errors.append(f"{label}: id must be a non-empty string")
        elif dataset_id in seen_dataset_ids:
            errors.append(f"{label}: duplicate dataset id {dataset_id}")
        else:
            seen_dataset_ids.add(dataset_id)
        if split not in {"train", "validation", "test"}:
            errors.append(f"{label}: split must be train, validation, or test")

        path, path_error = _safe_repo_path(repo_root, record.get("path"))
        if path_error:
            errors.append(f"{label}: {path_error}")
            continue
        assert path is not None
        if not path.is_file():
            errors.append(f"{label}: file does not exist: {record.get('path')}")
            continue
        _validate_digest(label=label, record=record, path=path, errors=errors)
        expected_bytes = record.get("bytes")
        if not isinstance(expected_bytes, int) or expected_bytes < 0:
            errors.append(f"{label}: bytes must be a non-negative integer")
        elif path.stat().st_size != expected_bytes:
            errors.append(
                f"{label}: byte-size mismatch (expected {expected_bytes}, "
                f"got {path.stat().st_size})"
            )

        expected_rows = record.get("rows")
        if not isinstance(expected_rows, int) or expected_rows < 0:
            errors.append(f"{label}: rows must be a non-negative integer")
            expected_rows = None

        count = 0
        local_ids: set[str] = set()
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    count += 1
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        errors.append(f"{label}:{line_number}: row must be a JSON object")
                        continue
                    row_id = row.get("id")
                    text = row.get("text")
                    source = row.get("source")
                    if not isinstance(row_id, str) or not row_id:
                        errors.append(f"{label}:{line_number}: id must be a non-empty string")
                    elif row_id in local_ids:
                        errors.append(f"{label}:{line_number}: duplicate id {row_id}")
                    else:
                        local_ids.add(row_id)
                        previous = global_ids.get(row_id)
                        if previous is not None:
                            errors.append(
                                f"{label}:{line_number}: id {row_id} also appears in {previous}"
                            )
                        else:
                            global_ids[row_id] = label
                    if not isinstance(text, str) or not text.strip():
                        errors.append(f"{label}:{line_number}: text must be a non-empty string")
                        continue
                    if source != record.get("source_field"):
                        errors.append(
                            f"{label}:{line_number}: source {source!r} does not match "
                            f"source_field {record.get('source_field')!r}"
                        )
                    text_digest = hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()
                    target = train_texts if split == "train" else evaluation_texts
                    target.setdefault(text_digest, f"{label}:{line_number}")
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            errors.append(f"{label}: cannot parse JSONL: {exc}")
            continue

        if expected_rows is not None and count != expected_rows:
            errors.append(f"{label}: row-count mismatch (expected {expected_rows}, got {count})")

    for text_digest in sorted(train_texts.keys() & evaluation_texts.keys()):
        if text_digest in allowed_text_digests:
            found_allowed_digests.add(text_digest)
        else:
            errors.append(
                "dataset leakage: normalized text appears in both "
                f"{train_texts[text_digest]} and {evaluation_texts[text_digest]}"
            )
    for text_digest in sorted(allowed_text_digests.keys() - found_allowed_digests):
        errors.append(
            "known_leakage entry does not match a current train/evaluation overlap: "
            f"{text_digest}"
        )


def _validate_artifacts(
    records: object,
    repo_root: Path,
    errors: list[str],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, dict[str, Mapping[str, Any]]]]:
    if not isinstance(records, dict) or not records:
        errors.append("artifacts must be a non-empty mapping")
        return {}, {}

    valid_records: dict[str, Mapping[str, Any]] = {}
    experiment_records: dict[str, dict[str, Mapping[str, Any]]] = {}
    for artifact_id, raw_record in records.items():
        label = f"artifacts.{artifact_id}"
        if not isinstance(artifact_id, str) or not artifact_id:
            errors.append("artifact keys must be non-empty strings")
            continue
        if not isinstance(raw_record, dict):
            errors.append(f"{label}: record must be a mapping")
            continue
        record = raw_record
        if not isinstance(record.get("kind"), str) or not record["kind"]:
            errors.append(f"{label}: kind must be a non-empty string")
        path, path_error = _safe_repo_path(repo_root, record.get("path"))
        if path_error:
            errors.append(f"{label}: {path_error}")
            continue
        assert path is not None
        if not path.is_file():
            errors.append(f"{label}: file does not exist: {record.get('path')}")
            continue
        _validate_digest(label=label, record=record, path=path, errors=errors)
        expected_bytes = record.get("bytes")
        if not isinstance(expected_bytes, int) or expected_bytes < 0:
            errors.append(f"{label}: bytes must be a non-negative integer")
        elif path.stat().st_size != expected_bytes:
            errors.append(
                f"{label}: byte-size mismatch (expected {expected_bytes}, "
                f"got {path.stat().st_size})"
            )
        valid_records[artifact_id] = record
        if path.suffix.lower() == ".json":
            try:
                with path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                    experiment_records[artifact_id] = {
                        item["experiment_id"]: item
                        for item in _iter_experiment_records(payload)
                    }
                    _validate_json_assertions(
                        label=label,
                        payload=payload,
                        assertions=record.get("assertions"),
                        errors=errors,
                    )
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                errors.append(f"{label}: cannot parse JSON artifact: {exc}")
    return valid_records, experiment_records


def _validate_runs(
    runs: object,
    naming: object,
    artifacts: Mapping[str, Mapping[str, Any]],
    artifact_experiment_records: Mapping[str, Mapping[str, Mapping[str, Any]]],
    errors: list[str],
) -> None:
    if not isinstance(runs, list) or not runs:
        errors.append("runs must be a non-empty list")
        return
    if not isinstance(naming, dict):
        errors.append("naming must be a mapping")
        return
    raw_pattern = naming.get("new_run_id_pattern")
    try:
        pattern = re.compile(raw_pattern) if isinstance(raw_pattern, str) else None
    except re.error as exc:
        errors.append(f"naming.new_run_id_pattern is invalid: {exc}")
        pattern = None
    if pattern is None:
        errors.append("naming.new_run_id_pattern must be a valid regex string")

    seen_ids: set[str] = set()
    for index, raw_run in enumerate(runs):
        label = f"runs[{index}]"
        if not isinstance(raw_run, dict):
            errors.append(f"{label}: run must be a mapping")
            continue
        run_id = raw_run.get("id")
        if not isinstance(run_id, str) or not run_id:
            errors.append(f"{label}: id must be a non-empty string")
            continue
        if run_id in seen_ids:
            errors.append(f"{label}: duplicate run id {run_id}")
        seen_ids.add(run_id)
        id_style = raw_run.get("id_style")
        if id_style not in {"historical", "revision"}:
            errors.append(f"{label}: id_style must be historical or revision")
        elif id_style == "revision" and pattern is not None and not pattern.fullmatch(run_id):
            errors.append(f"{label}: revision run id does not match naming policy: {run_id}")
        for field in ("status", "model_id", "strategy", "initialization", "epochs"):
            if raw_run.get(field) in (None, ""):
                errors.append(f"{label}: missing required field {field}")
        if raw_run.get("strategy") not in {"original", "replacement", "extension"}:
            errors.append(f"{label}: strategy must be original, replacement, or extension")
        if raw_run.get("initialization") not in {"random", "mean_subword", "preserved"}:
            errors.append(f"{label}: unsupported initialization")
        vocab_size = raw_run.get("vocab_size")
        if not isinstance(vocab_size, int) or vocab_size <= 0:
            errors.append(f"{label}: vocab_size must be a positive integer")
        seed = raw_run.get("seed")
        if not isinstance(seed, int) and not _is_not_recorded(seed):
            errors.append(f"{label}: seed must be an integer or explicit not_recorded marker")
        artifact_id = raw_run.get("result_artifact")
        if not isinstance(artifact_id, str) or artifact_id not in artifacts:
            errors.append(f"{label}: result_artifact must reference a declared artifact")
        elif raw_run.get("status") == "completed":
            records = artifact_experiment_records.get(artifact_id, {})
            if run_id not in records:
                errors.append(
                    f"{label}: completed run {run_id} not found in artifact {artifact_id}"
                )
            else:
                result_model_id = records[run_id].get("model_id")
                if result_model_id != raw_run.get("model_id"):
                    errors.append(
                        f"{label}: model_id does not match result artifact "
                        f"({raw_run.get('model_id')!r} != {result_model_id!r})"
                    )


def validate_revision_manifest(manifest_path: Path, repo_root: Path | None = None) -> list[str]:
    """Return all validation errors for a revision manifest."""
    errors: list[str] = []
    try:
        manifest = load_revision_manifest(manifest_path)
    except (OSError, UnicodeError, yaml.YAMLError, ValueError) as exc:
        return [f"cannot load manifest: {exc}"]

    missing = sorted(REQUIRED_SECTIONS - manifest.keys())
    if missing:
        errors.append(f"missing required sections: {', '.join(missing)}")
    if manifest.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    for section in ("project", "environments", "protocols", "planned_revision", "run_log"):
        if section in manifest and not isinstance(manifest[section], dict):
            errors.append(f"{section} must be a mapping")
    if "limitations" in manifest and not isinstance(manifest["limitations"], list):
        errors.append("limitations must be a list")
    active_protocol = manifest.get("active_protocol")
    if active_protocol is not None:
        if not isinstance(active_protocol, dict):
            errors.append("active_protocol must be a mapping")
        elif active_protocol.get("zero_active_leakage_exceptions") is True and manifest.get(
            "known_leakage"
        ):
            errors.append(
                "active_protocol requires zero leakage exceptions but known_leakage is not empty"
            )
    repository = manifest.get("repository")
    if not isinstance(repository, dict):
        errors.append("repository must be a mapping")
    else:
        commit = repository.get("baseline_commit")
        if not isinstance(commit, str) or not GIT_COMMIT_RE.fullmatch(commit):
            errors.append("repository.baseline_commit must be a 40-character lowercase Git SHA")

    root = repo_root or manifest_path.resolve().parents[1]
    _validate_dataset_records(
        manifest.get("datasets"),
        manifest.get("known_leakage"),
        root,
        errors,
    )
    artifacts, experiment_ids = _validate_artifacts(manifest.get("artifacts"), root, errors)
    _validate_runs(
        manifest.get("runs"),
        manifest.get("naming"),
        artifacts,
        experiment_ids,
        errors,
    )
    return errors
