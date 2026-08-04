"""Build a deterministic audit of the historical domain router."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import sklearn
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from akan_bpe.classifier import DOMAIN_ASR, DOMAIN_TTS, load_classifier, load_training_data
from akan_bpe.datasets import load_jsonl_samples


def _jsonable_params(params: dict[str, Any]) -> dict[str, Any]:
    """Keep public scalar estimator parameters in a JSON-safe form."""
    result: dict[str, Any] = {}
    for key, value in params.items():
        if value is None or isinstance(value, (bool, int, float, str)):
            result[key] = value
        elif isinstance(value, tuple):
            result[key] = list(value)
    return result


def _metrics(labels: list[int], predictions: list[int]) -> dict[str, Any]:
    report = classification_report(
        labels,
        predictions,
        labels=[0, 1],
        target_names=[DOMAIN_ASR, DOMAIN_TTS],
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": report["accuracy"],
        "per_class": {
            domain: {
                "precision": report[domain]["precision"],
                "recall": report[domain]["recall"],
                "f1": report[domain]["f1-score"],
                "support": int(report[domain]["support"]),
            }
            for domain in (DOMAIN_ASR, DOMAIN_TTS)
        },
        "macro_average": {
            "precision": report["macro avg"]["precision"],
            "recall": report["macro avg"]["recall"],
            "f1": report["macro avg"]["f1-score"],
        },
        "confusion_matrix": {
            "label_order": [DOMAIN_ASR, DOMAIN_TTS],
            "rows_true_columns_predicted": confusion_matrix(
                labels, predictions, labels=[0, 1]
            ).tolist(),
        },
    }


def _load_texts(path: Path) -> list[str]:
    return [sample.text for sample in load_jsonl_samples(path)]


def _read_router_counts(path: Path, true_label: int) -> tuple[list[int], list[int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    decisions = payload["routing_decisions"]
    labels = [true_label] * int(payload["total_samples"])
    predictions = [0] * int(decisions[DOMAIN_ASR]) + [1] * int(decisions[DOMAIN_TTS])
    return labels, predictions


def build_router_audit(
    *,
    classifier_path: Path,
    asr_train_path: Path,
    tts_train_path: Path,
    asr_test_path: Path,
    tts_test_path: Path,
    heuristic_asr_result_path: Path,
    heuristic_tts_result_path: Path,
) -> dict[str, Any]:
    """Audit the frozen classifier without retraining or rewriting it."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", InconsistentVersionWarning)
        classifier = load_classifier(classifier_path)

    texts, labels = load_training_data(str(asr_train_path), str(tts_train_path))
    _, heldout_texts, _, heldout_labels = train_test_split(
        texts,
        labels,
        test_size=0.20,
        random_state=42,
        stratify=labels,
    )
    heldout_predictions = classifier.predict(heldout_texts).tolist()

    asr_texts = _load_texts(asr_test_path)
    tts_texts = _load_texts(tts_test_path)
    external_labels = [0] * len(asr_texts) + [1] * len(tts_texts)
    external_predictions = classifier.predict(asr_texts + tts_texts).tolist()

    heuristic_asr_labels, heuristic_asr_predictions = _read_router_counts(
        heuristic_asr_result_path, 0
    )
    heuristic_tts_labels, heuristic_tts_predictions = _read_router_counts(
        heuristic_tts_result_path, 1
    )

    vectorizer = classifier.named_steps["vectorizer"]
    model = classifier.named_steps["classifier"]
    return {
        "experiment_id": "router-audit-revision-v2",
        "status": "complete_demoted_to_secondary_analysis",
        "decision": {
            "paper_role": "secondary_analysis",
            "central_contribution": False,
            "reason": (
                "Evaluation predicts source-corpus identity on separately collected ASR and formal "
                "text. Without an ambiguous or mixed-domain challenge set, near-perfect accuracy "
                "does not establish robust routing under realistic domain ambiguity."
            ),
        },
        "protocol": {
            "task": "binary source-domain classification",
            "labels": {"0": DOMAIN_ASR, "1": DOMAIN_TTS},
            "training_sources": {
                DOMAIN_ASR: str(asr_train_path).replace("\\", "/"),
                DOMAIN_TTS: str(tts_train_path).replace("\\", "/"),
            },
            "split": {
                "method": "stratified_random_holdout",
                "test_fraction": 0.20,
                "random_state": 42,
                "total": len(labels),
                "train": len(labels) - len(heldout_labels),
                "test": len(heldout_labels),
                "class_totals": {
                    DOMAIN_ASR: labels.count(0),
                    DOMAIN_TTS: labels.count(1),
                },
                "test_class_totals": {
                    DOMAIN_ASR: heldout_labels.count(0),
                    DOMAIN_TTS: heldout_labels.count(1),
                },
            },
            "vectorizer": {
                "type": type(vectorizer).__name__,
                "parameters": _jsonable_params(vectorizer.get_params(deep=False)),
            },
            "classifier": {
                "type": type(model).__name__,
                "parameters": _jsonable_params(model.get_params(deep=False)),
            },
        },
        "heldout_source_classification": _metrics(heldout_labels, heldout_predictions),
        "external_corpus_classification": {
            "test_files": {
                DOMAIN_ASR: str(asr_test_path).replace("\\", "/"),
                DOMAIN_TTS: str(tts_test_path).replace("\\", "/"),
            },
            "ml": _metrics(external_labels, external_predictions),
            "heuristic": _metrics(
                heuristic_asr_labels + heuristic_tts_labels,
                heuristic_asr_predictions + heuristic_tts_predictions,
            ),
        },
        "limitations": {
            "ambiguous_challenge_set_available": False,
            "mixed_domain_ground_truth_evaluated": False,
            "routing_latency_measured": False,
            "tokenizer_switching_overhead_measured": False,
            "model_checkpoint_compatibility": (
                "Separate replacement tokenizers require separate compatible embedding interfaces; "
                "the router is not a drop-in tokenizer switch for one frozen checkpoint."
            ),
            "pickle_runtime": {
                "runtime_sklearn_version": sklearn.__version__,
                "inconsistent_version_warning_observed": any(
                    isinstance(item.message, InconsistentVersionWarning) for item in caught
                ),
            },
        },
    }
