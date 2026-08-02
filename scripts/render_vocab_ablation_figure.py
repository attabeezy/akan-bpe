#!/usr/bin/env python3
"""Render SVG and PNG ablation figures from the reviewed aggregate JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from akan_bpe.vocab_ablation import save_fertility_figure


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/vocab_ablation_results.json"),
    )
    parser.add_argument(
        "--output-svg",
        type=Path,
        default=Path("results/vocab_ablation_fertility.svg"),
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("results/vocab_ablation_fertility.png"),
    )
    args = parser.parse_args()
    payload = cast(dict[str, Any], json.loads(args.input.read_text(encoding="utf-8")))
    save_fertility_figure(payload, svg_path=args.output_svg, png_path=args.output_png)
    print(f"SVG written to {args.output_svg}")
    print(f"PNG written to {args.output_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
