"""Vocabulary-size ablation training, metrics, uncertainty, and figure helpers."""

from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Any

import numpy as np

from akan_bpe.datasets import load_jsonl_samples, samples_to_texts
from akan_bpe.revision_manifest import sha256_file
from akan_bpe.tokenizers import (
    DEFAULT_SPECIAL_TOKENS,
    train_bpe_tokenizer,
)

BASELINE_TOKENIZERS = {
    "xlm_roberta_base": "xlm-roberta-base",
    "bert_base_multilingual_cased": "bert-base-multilingual-cased",
    "mt5_base": "google/mt5-base",
}
TARGET_MODELS = {
    "Qwen/Qwen3-0.6B-Base": 1024,
    "google/gemma-3-1b-pt": 1152,
    "Qwen/Qwen3-1.7B-Base": 2048,
    "meta-llama/Llama-3.2-1B": 2048,
    "CohereLabs/tiny-aya-base": 2048,
}


def build_balanced_training_texts(input_paths: list[Path]) -> tuple[list[str], dict[str, int]]:
    """Upsample every input corpus to the largest corpus and return a fixed-order mixture."""
    if len(input_paths) < 2:
        raise ValueError("Balanced tokenizer training requires at least two input corpora.")
    per_file_samples = [load_jsonl_samples(path) for path in input_paths]
    if any(not samples for samples in per_file_samples):
        raise ValueError("Every balanced training corpus must contain at least one valid row.")
    max_count = max(len(samples) for samples in per_file_samples)
    balanced_samples = []
    original_counts: dict[str, int] = {}
    for path, samples in zip(input_paths, per_file_samples):
        original_counts[path.as_posix()] = len(samples)
        repeat_times = math.ceil(max_count / len(samples))
        balanced_samples.extend((samples * repeat_times)[:max_count])
    return samples_to_texts(balanced_samples), original_counts


def train_vocab_variants(
    *,
    training_texts: list[str],
    vocab_sizes: list[int],
    output_dir: Path,
    input_paths: list[Path],
    original_counts: dict[str, int],
) -> list[dict[str, object]]:
    """Train all requested BPE variants with one frozen corpus and configuration."""
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict[str, object]] = []
    for vocab_size in vocab_sizes:
        output_path = output_dir / f"mixed_bpe_v{vocab_size}.json"
        info = train_bpe_tokenizer(
            texts=training_texts,
            output_path=output_path,
            vocab_size=vocab_size,
            name=f"mixed_bpe_v{vocab_size}",
        )
        artifacts.append(
            {
                "target_vocab_size": vocab_size,
                "actual_vocab_size": info.vocab_size,
                "path": output_path.as_posix(),
                "bytes": output_path.stat().st_size,
                "sha256": sha256_file(output_path),
                "training_rows": len(training_texts),
                "input_files": [path.as_posix() for path in input_paths],
                "original_row_counts": original_counts,
                "balance_rule": "upsample_each_corpus_to_largest_then_concatenate",
                "algorithm": "BPE",
                "normalization": "none",
                "pre_tokenizer": "Whitespace",
                "minimum_frequency": 0,
                "special_tokens": DEFAULT_SPECIAL_TOKENS,
            }
        )
    return artifacts


def _tokenizer_vocab_size(tokenizer: Any) -> int:
    vocab = tokenizer.get_vocab()
    return len(vocab)


def evaluate_tokenizer(
    *,
    tokenizer: Any,
    tokenizer_name: str,
    tokenizer_reference: str,
    texts: list[str],
    source_file: str,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    """Evaluate one tokenizer and retain paired per-example counts for bootstrap tests."""
    token_counts: list[int] = []
    word_counts: list[int] = []
    used_ids: set[int] = set()
    unknown_count = 0
    special_count = 0
    unknown_id = getattr(tokenizer, "unk_token_id", None)
    special_ids = set(getattr(tokenizer, "all_special_ids", []))

    for text in texts:
        token_ids = tokenizer.encode(text)
        if not isinstance(token_ids, list):
            token_ids = token_ids.ids
        ids = [int(token_id) for token_id in token_ids]
        words = len(text.split())
        token_counts.append(len(ids))
        word_counts.append(words)
        used_ids.update(ids)
        if unknown_id is not None:
            unknown_count += sum(token_id == unknown_id for token_id in ids)
        special_count += sum(token_id in special_ids for token_id in ids)

    token_array = np.asarray(token_counts, dtype=np.int64)
    word_array = np.asarray(word_counts, dtype=np.int64)
    total_tokens = int(token_array.sum())
    total_words = int(word_array.sum())
    vocab_size = _tokenizer_vocab_size(tokenizer)
    total_bytes = sum(len(text.encode("utf-8")) for text in texts)
    total_characters = sum(len(text) for text in texts)
    per_sample_fertility = np.divide(
        token_array,
        word_array,
        out=np.zeros_like(token_array, dtype=np.float64),
        where=word_array > 0,
    )
    metrics: dict[str, object] = {
        "tokenizer_name": tokenizer_name,
        "tokenizer_reference": tokenizer_reference,
        "source_file": source_file,
        "num_samples": len(texts),
        "vocab_size": vocab_size,
        "fertility": total_tokens / total_words if total_words else 0.0,
        "fertility_sample_std": (
            float(np.std(per_sample_fertility, ddof=1))
            if len(per_sample_fertility) > 1
            else 0.0
        ),
        "total_tokens": total_tokens,
        "total_words": total_words,
        "sequence_length": {
            "mean": float(np.mean(token_array)) if len(token_array) else 0.0,
            "median": float(np.median(token_array)) if len(token_array) else 0.0,
            "p90": float(np.percentile(token_array, 90)) if len(token_array) else 0.0,
            "p95": float(np.percentile(token_array, 95)) if len(token_array) else 0.0,
            "max": int(token_array.max()) if len(token_array) else 0,
        },
        "vocabulary_utilization": {
            "unique_token_ids": len(used_ids),
            "ratio": len(used_ids) / vocab_size if vocab_size else 0.0,
            "percent": len(used_ids) / vocab_size * 100 if vocab_size else 0.0,
        },
        "bytes_per_token": total_bytes / total_tokens if total_tokens else 0.0,
        "characters_per_token": total_characters / total_tokens if total_tokens else 0.0,
        "unknown_tokens": {
            "count": unknown_count,
            "ratio": unknown_count / total_tokens if total_tokens else 0.0,
        },
        "special_tokens": {
            "count": special_count,
            "ratio": special_count / total_tokens if total_tokens else 0.0,
        },
    }
    return metrics, token_array, word_array


def paired_bootstrap_fertility_difference(
    *,
    candidate_token_counts: np.ndarray,
    baseline_token_counts: np.ndarray,
    word_counts: np.ndarray,
    resamples: int,
    seed: int,
) -> dict[str, float | int]:
    """Bootstrap aggregate fertility(candidate)-fertility(baseline) over paired rows."""
    if not (
        len(candidate_token_counts) == len(baseline_token_counts) == len(word_counts)
        and len(word_counts) > 0
    ):
        raise ValueError("Paired bootstrap inputs must have the same non-zero length.")
    if resamples <= 0:
        raise ValueError("resamples must be positive.")
    rng = np.random.default_rng(seed)
    differences = np.empty(resamples, dtype=np.float64)
    row_count = len(word_counts)
    for index in range(resamples):
        sampled = rng.integers(0, row_count, size=row_count)
        sampled_words = int(word_counts[sampled].sum())
        candidate_fertility = float(candidate_token_counts[sampled].sum()) / sampled_words
        baseline_fertility = float(baseline_token_counts[sampled].sum()) / sampled_words
        differences[index] = candidate_fertility - baseline_fertility
    observed = float(candidate_token_counts.sum() / word_counts.sum()) - float(
        baseline_token_counts.sum() / word_counts.sum()
    )
    return {
        "difference_candidate_minus_baseline": observed,
        "confidence_level": 0.95,
        "lower": float(np.percentile(differences, 2.5)),
        "upper": float(np.percentile(differences, 97.5)),
        "resamples": resamples,
        "seed": seed,
    }


def embedding_costs(vocab_size: int) -> dict[str, dict[str, int | float]]:
    """Compute untied input-plus-output embedding cost for each target hidden size."""
    costs: dict[str, dict[str, int | float]] = {}
    for model_id, hidden_size in TARGET_MODELS.items():
        parameters = 2 * vocab_size * hidden_size
        costs[model_id] = {
            "hidden_size": hidden_size,
            "input_plus_output_parameters": parameters,
            "fp16_mib": parameters * 2 / (1024**2),
            "fp32_mib": parameters * 4 / (1024**2),
        }
    return costs


def select_operating_point(
    rows: list[dict[str, Any]],
    relative_plateau_tolerance: float = 0.01,
) -> dict[str, object]:
    """Choose the smallest vocabulary within the fixed relative plateau on both domains."""
    if not rows:
        raise ValueError("Operating-point selection requires at least one result row.")
    if relative_plateau_tolerance < 0:
        raise ValueError("Plateau tolerance cannot be negative.")
    best_asr = min(float(row["asr_fertility"]) for row in rows)
    best_formal = min(float(row["formal_fertility"]) for row in rows)
    qualifying = [
        row
        for row in rows
        if float(row["asr_fertility"]) <= best_asr * (1 + relative_plateau_tolerance)
        and float(row["formal_fertility"]) <= best_formal * (1 + relative_plateau_tolerance)
    ]
    selected = min(qualifying, key=lambda row: int(row["vocab_size"]))
    selected_vocab_size = int(selected["vocab_size"])
    max_tested_vocab_size = max(int(row["vocab_size"]) for row in rows)
    status = (
        "boundary_selected_no_earlier_plateau"
        if selected_vocab_size == max_tested_vocab_size and len(qualifying) == 1
        else "plateau_observed"
    )
    return {
        "rule": (
            "Select the smallest vocabulary whose fertility is within 1% relative of "
            "the best observed fertility in both ASR and formal regimes."
        ),
        "relative_plateau_tolerance": relative_plateau_tolerance,
        "best_asr_fertility": best_asr,
        "best_formal_fertility": best_formal,
        "qualifying_vocab_sizes": [int(row["vocab_size"]) for row in qualifying],
        "selected_vocab_size": selected_vocab_size,
        "status": status,
        "boundary_caveat": (
            "Only the largest tested vocabulary meets the fixed threshold, so this is "
            "a boundary selection rather than evidence that the curve has fully plateaued."
            if status == "boundary_selected_no_earlier_plateau"
            else None
        ),
        "model_quality_override": (
            "A later controlled model-quality result may override this tokenizer-only choice."
        ),
    }


def render_tradeoff_markdown(payload: dict[str, Any]) -> str:
    """Render the compact ablation trade-off table from aggregate results."""
    lines = [
        "# Vocabulary-Size Ablation",
        "",
        "| Vocab | ASR fertility | ASR p95 | ASR util. | Formal fertility | "
        "Formal p95 | Formal util. | Tokenizer MiB | Qwen 0.6B interface MiB (FP16) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["tradeoff_rows"]:
        lines.append(
            f"| {row['vocab_size']:,} | {row['asr_fertility']:.6f} | "
            f"{row['asr_p95']:.1f} | {row['asr_utilization_percent']:.2f}% | "
            f"{row['formal_fertility']:.6f} | {row['formal_p95']:.1f} | "
            f"{row['formal_utilization_percent']:.2f}% | "
            f"{row['tokenizer_size_mib']:.3f} | "
            f"{row['embedding_costs']['Qwen/Qwen3-0.6B-Base']['fp16_mib']:.2f} |"
        )
    operating = payload["operating_point"]
    lines.extend(
        [
            "",
            "## Operating point",
            "",
            f"Selected vocabulary: **{operating['selected_vocab_size']:,}**",
            "",
            f"Rule: {operating['rule']}",
            f"Status: **{operating['status']}**",
            *(
                ["", f"Caveat: {operating['boundary_caveat']}"]
                if operating["boundary_caveat"]
                else []
            ),
            "",
            "Lower fertility is better. Interface memory counts untied input embeddings and "
            "the output language-model head, matching the current replacement pipeline.",
            "",
        ]
    )
    return "\n".join(lines)


def render_fertility_svg(payload: dict[str, Any], width: int = 1000, height: int = 650) -> str:
    """Render a dependency-free, report-ready SVG fertility curve."""
    rows = payload["tradeoff_rows"]
    left, right, top, bottom = 100, width - 60, 110, height - 100
    plot_width, plot_height = right - left, bottom - top
    max_value = max(
        max(float(row["asr_fertility"]), float(row["formal_fertility"])) for row in rows
    )
    y_max = math.ceil(max_value * 10) / 10 + 0.1
    tick_count = 7

    def x_position(index: int) -> float:
        return left + index * plot_width / max(len(rows) - 1, 1)

    def y_position(value: float) -> float:
        return bottom - value / y_max * plot_height

    series = (
        ("ASR v2", "asr_fertility", "#1f5a94", "none"),
        ("Formal Twi", "formal_fertility", "#b27700", "7 5"),
    )
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#20252b}'
        '.mono{font-family:Consolas,monospace}</style>',
        '<text x="60" y="45" font-size="26" font-weight="700">Fertility by vocabulary size</text>',
        '<text x="60" y="76" font-size="15" fill="#59636e">'
        "Balanced mixed BPE; lower is better; ASR n=1,010, formal n=2,500</text>",
    ]
    for tick in range(tick_count + 1):
        value = y_max * tick / tick_count
        y = y_position(value)
        elements.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" '
            'stroke="#dce2e8" stroke-width="1"/>'
        )
        elements.append(
            f'<text class="mono" x="{left - 14}" y="{y + 5:.1f}" '
            f'font-size="13" text-anchor="end">{value:.1f}</text>'
        )
    elements.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" '
            'stroke="#59636e" stroke-width="1.5"/>',
            f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" '
            'stroke="#59636e" stroke-width="1.5"/>',
        ]
    )
    for index, row in enumerate(rows):
        x = x_position(index)
        elements.append(
            f'<text class="mono" x="{x:.1f}" y="{bottom + 30}" font-size="14" '
            f'text-anchor="middle">{int(row["vocab_size"]) // 1000}K</text>'
        )
    elements.append(
        f'<text x="{(left + right) / 2:.1f}" y="{height - 35}" '
        'font-size="15" text-anchor="middle">Vocabulary size</text>'
    )
    elements.append(
        f'<text x="28" y="{(top + bottom) / 2:.1f}" font-size="15" '
        f'text-anchor="middle" transform="rotate(-90 28 {(top + bottom) / 2:.1f})">'
        "Tokens per whitespace word</text>"
    )
    for series_index, (label, field, color, dash) in enumerate(series):
        points = " ".join(
            f"{x_position(index):.1f},{y_position(float(row[field])):.1f}"
            for index, row in enumerate(rows)
        )
        dash_attribute = f' stroke-dasharray="{dash}"' if dash != "none" else ""
        elements.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"'
            f'{dash_attribute}/>'
        )
        for index, row in enumerate(rows):
            x = x_position(index)
            y = y_position(float(row[field]))
            fill = color if series_index == 0 else "#fbfcfe"
            elements.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{fill}" '
                f'stroke="{color}" stroke-width="3"/>'
            )
            label_y = y - 13 if series_index == 0 else y + 24
            elements.append(
                f'<text class="mono" x="{x:.1f}" y="{label_y:.1f}" font-size="12" '
                f'text-anchor="middle" fill="{color}">{float(row[field]):.3f}</text>'
            )
        legend_x = left + series_index * 180
        elements.append(
            f'<line x1="{legend_x}" y1="92" x2="{legend_x + 35}" y2="92" '
            f'stroke="{color}" stroke-width="3"{dash_attribute}/>'
        )
        elements.append(
            f'<text x="{legend_x + 44}" y="97" font-size="14">{html.escape(label)}</text>'
        )
    elements.append("</svg>")
    return "\n".join(elements)


def save_fertility_figure(
    payload: dict[str, Any],
    *,
    svg_path: Path,
    png_path: Path,
) -> None:
    """Export the fertility curve through one Matplotlib figure to SVG and PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = payload["tradeoff_rows"]
    vocab_sizes = [int(row["vocab_size"]) for row in rows]
    labels = [f"{size // 1000}K" for size in vocab_sizes]
    asr_values = [float(row["asr_fertility"]) for row in rows]
    formal_values = [float(row["formal_fertility"]) for row in rows]
    matplotlib.rcParams["svg.fonttype"] = "none"
    matplotlib.rcParams["svg.hashsalt"] = "akan-bpe-revision-v2"
    matplotlib.rcParams["font.family"] = "DejaVu Sans"

    figure, axis = plt.subplots(figsize=(10, 6.5), facecolor="#fbfcfe")
    axis.set_facecolor("#fbfcfe")
    axis.plot(
        labels,
        asr_values,
        color="#1f5a94",
        linewidth=2.8,
        marker="o",
        markersize=8,
        markerfacecolor="#1f5a94",
        label="ASR v2",
    )
    axis.plot(
        labels,
        formal_values,
        color="#b27700",
        linewidth=2.8,
        linestyle="--",
        marker="o",
        markersize=8,
        markerfacecolor="#fbfcfe",
        markeredgewidth=2.2,
        label="Formal Twi",
    )
    for index, value in enumerate(asr_values):
        axis.annotate(
            f"{value:.3f}",
            (index, value),
            xytext=(0, -18),
            textcoords="offset points",
            ha="center",
            color="#1f5a94",
            fontsize=9,
            fontfamily="DejaVu Sans Mono",
        )
    for index, value in enumerate(formal_values):
        axis.annotate(
            f"{value:.3f}",
            (index, value),
            xytext=(0, 11),
            textcoords="offset points",
            ha="center",
            color="#8c5c00",
            fontsize=9,
            fontfamily="DejaVu Sans Mono",
        )
    axis.set_ylim(0, max(*asr_values, *formal_values) * 1.13)
    axis.set_xlabel("Vocabulary size", fontsize=11)
    axis.set_ylabel("Tokens per whitespace word", fontsize=11)
    axis.grid(axis="y", color="#dce2e8", linewidth=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#59636e")
    axis.spines["bottom"].set_color("#59636e")
    axis.tick_params(colors="#37414b")
    axis.legend(loc="upper right", frameon=False, ncol=2)
    figure.suptitle(
        "Fertility by vocabulary size",
        x=0.08,
        y=0.97,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color="#20252b",
    )
    figure.text(
        0.08,
        0.92,
        "Balanced mixed BPE; lower is better; ASR n=1,010, formal n=2,500",
        ha="left",
        fontsize=10,
        color="#59636e",
    )
    figure.subplots_adjust(left=0.11, right=0.96, top=0.84, bottom=0.12)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        svg_path,
        format="svg",
        facecolor=figure.get_facecolor(),
        metadata={"Date": None, "Creator": "akan-bpe revision-v2"},
    )
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    figure.savefig(
        png_path,
        format="png",
        dpi=200,
        facecolor=figure.get_facecolor(),
        metadata={"Software": "akan-bpe revision-v2"},
    )
    plt.close(figure)
