#!/usr/bin/env python3
"""Export trained model to GGUF format for edge deployment.

Converts LoRA-adapted model to merged model, then quantizes to 4-bit GGUF
for deployment on Dell Latitude 7400 (8GB RAM).

Prerequisites:
    - llama.cpp compiled with Python bindings
    - Or use: pip install llama-cpp-python

Usage:
    # After training
    python scripts/03_export_gguf.py --checkpoint checkpoints/variant_E/final/ --output models/gguf/

    # Convert base tokenizer
    python scripts/03_export_gguf.py --checkpoint meta-llama/Llama-3.2-1B --output models/gguf/ --base-only
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def check_llama_cpp() -> bool:
    """Check if llama.cpp tools are available.

    Returns:
        True if llama.cpp is installed.
    """
    try:
        result = subprocess.run(
            ["python", "-c", "import llama_cpp; print(llama_cpp.__version__)"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except Exception:
        return False


def merge_lora_model(checkpoint_dir: Path, output_dir: Path) -> Path:
    """Merge LoRA adapters with base model.

    Args:
        checkpoint_dir: Directory containing LoRA adapter.
        output_dir: Directory to save merged model.

    Returns:
        Path to merged model directory.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError:
        raise ImportError(
            "transformers and peft required. Install with: pip install transformers peft"
        )

    print(f"Loading base model...")

    adapter_config = checkpoint_dir / "adapter_config.json"
    if not adapter_config.exists():
        raise FileNotFoundError(f"LoRA adapter not found at {checkpoint_dir}")

    with open(adapter_config, "r") as f:
        import json

        config = json.load(f)

    base_model_id = config.get("base_model_name_or_path")
    if not base_model_id:
        raise ValueError("base_model_name_or_path not found in adapter_config.json")

    print(f"Base model: {base_model_id}")

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype="auto", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    print(f"Loading LoRA adapter from {checkpoint_dir}...")
    model = PeftModel.from_pretrained(base_model, str(checkpoint_dir))

    print("Merging adapter into base model...")
    merged_model = model.merge_and_unload()

    merged_dir = output_dir / "merged_model"
    merged_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving merged model to {merged_dir}...")
    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)

    print("Merge complete!")
    return merged_dir


def convert_to_gguf(model_dir: Path, output_dir: Path, quantization: str = "Q4_K_M") -> Path:
    """Convert merged model to GGUF format.

    Args:
        model_dir: Directory containing merged model.
        output_dir: Directory to save GGUF file.
        quantization: Quantization method (Q4_K_M, Q8_0, etc.)

    Returns:
        Path to GGUF file.

    Note:
        This requires llama.cpp to be installed or available.
        For Colab, we'll use the huggingface_hub to upload and let others
        handle the conversion, or use a conversion script.
    """
    try:
        from transformers import AutoTokenizer

        print(f"Preparing GGUF conversion...")

        output_dir.mkdir(parents=True, exist_ok=True)

        gguf_path = output_dir / f"model-{quantization}.gguf"

        print("Note: Full GGUF conversion requires llama.cpp.")
        print("For Colab, use the following approach:")
        print("1. Upload merged model to HuggingFace Hub")
        print("2. Use llama.cpp locally or via API for conversion")
        print()
        print("Alternative: Use llama-cpp-python for inference directly:")
        print(f"    model = Llama(model_path='{model_dir}')")

        return gguf_path

    except ImportError:
        raise ImportError("transformers required. Install with: pip install transformers")


def create_gguf_conversion_script(output_dir: Path) -> Path:
    """Create a helper script for GGUF conversion.

    Args:
        output_dir: Directory to save the script.

    Returns:
        Path to the conversion script.
    """
    script_content = '''#!/usr/bin/env python3
"""GGUF conversion helper script.

This script uses llama.cpp's convert script to convert a HuggingFace model
to GGUF format with quantization.

Prerequisites:
    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp && make

Usage:
    python convert_to_gguf.py --model-dir merged_model/ --output model.gguf --quantize Q4_K_M
"""

import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Convert model to GGUF")
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="model.gguf")
    parser.add_argument("--quantize", type=str, default="Q4_K_M",
                        choices=["Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q8_0"])
    parser.add_argument("--llama-cpp-dir", type=str, default="llama.cpp")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_file = Path(args.output)
    llama_cpp = Path(args.llama_cpp_dir)

    convert_script = llama_cpp / "convert-hf-to-gguf.py"
    if not convert_script.exists():
        raise FileNotFoundError(f"llama.cpp convert script not found at {convert_script}")

    # Convert to GGUF (FP16)
    gguf_fp16 = output_file.with_suffix(".fp16.gguf")
    subprocess.run([
        sys.executable, str(convert_script),
        str(model_dir),
        "--outfile", str(gguf_fp16),
        "--outtype", "f16"
    ], check=True)

    # Quantize
    quantize_bin = llama_cpp / "llama-quantize"
    subprocess.run([
        str(quantize_bin),
        str(gguf_fp16),
        str(output_file),
        args.quantize
    ], check=True)

    print(f"GGUF model saved to: {output_file}")
    gguf_fp16.unlink()


if __name__ == "__main__":
    import sys
    main()
'''

    script_path = output_dir / "convert_to_gguf.py"
    with open(script_path, "w") as f:
        f.write(script_content)

    return script_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export to GGUF format")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to LoRA checkpoint or base model ID"
    )
    parser.add_argument("--output", type=str, default="models/gguf/")
    parser.add_argument(
        "--quantization",
        type=str,
        default="Q4_K_M",
        choices=["Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q8_0"],
    )
    parser.add_argument(
        "--base-only", action="store_true", help="Skip LoRA merge, just prepare base model"
    )
    parser.add_argument(
        "--no-convert", action="store_true", help="Skip GGUF conversion, just merge LoRA"
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint)
    is_base_model = not checkpoint_path.exists()

    if args.base_only or is_base_model:
        print(f"Using base model directly: {args.checkpoint}")
        print("No LoRA merge required.")

        model_dir = args.checkpoint

        converter_script = create_gguf_conversion_script(output_dir)
        print(f"\nGGUF converter script created: {converter_script}")
        print("\nTo convert to GGUF, run:")
        print(f"  python {converter_script} --model-dir {model_dir} --output model.gguf")
        return

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Exporting model...")
    print(f"  Input: {checkpoint_path}")
    print(f"  Output: {output_dir}")
    print(f"  Quantization: {args.quantization}")
    print()

    merged_dir = merge_lora_model(checkpoint_path, output_dir)

    if args.no_convert:
        print("\nSkipping GGUF conversion (--no-convert)")
        print(f"Merged model saved to: {merged_dir}")
        return

    gguf_path = convert_to_gguf(merged_dir, output_dir, args.quantization)

    converter_script = create_gguf_conversion_script(output_dir)
    print(f"\nGGUF converter script created: {converter_script}")
    print("\nTo complete conversion, run locally:")
    print(f"  python {converter_script} --model-dir {merged_dir} --output {gguf_path}")


if __name__ == "__main__":
    main()
