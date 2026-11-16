"""Faz merge de adaptador LoRA com o modelo base, converte para GGUF e quantiza (Q4_K_M)."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common import BASE_MODEL, DIR_ADAPTERS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--adapter-dir", default=str(DIR_ADAPTERS / "lora_arborgrid_r16"))
    parser.add_argument("--out-gguf", default=str(ROOT / "models" / "arborgrid-qwen3-1.7b-q4_k_m.gguf"))
    parser.add_argument("--quant", default="Q4_K_M")
    args = parser.parse_args()

    adapter_path = Path(args.adapter_dir)
    out_gguf = Path(args.out_gguf)
    out_gguf.parent.mkdir(parents=True, exist_ok=True)

    tag = out_gguf.stem.replace(".", "_")
    merged_dir = ROOT / f"checkpoints_merged_{tag}"
    shutil.rmtree(merged_dir, ignore_errors=True)
    merged_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Carregando {args.base_model} + {adapter_path} para merge em float16 (CPU)...")
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
    base = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.float16, device_map="cpu")
    peft_model = PeftModel.from_pretrained(base, str(adapter_path))
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))

    f16_gguf = merged_dir / "model-f16.gguf"
    converter = ROOT / "scripts" / "vendor" / "llamacpp_conv" / "convert_hf_to_gguf.py"
    print(f"[2/3] Convertendo HuggingFace -> F16 GGUF em {f16_gguf}...")
    subprocess.run(
        [sys.executable, str(converter), str(merged_dir), "--outfile", str(f16_gguf), "--outtype", "f16"],
        check=True,
    )

    if args.quant.upper() == "F16":
        print(f"[3/3] Mantendo precisão original 16 bits -> {out_gguf}...")
        shutil.move(str(f16_gguf), str(out_gguf))
    else:
        print(f"[3/3] Quantizando para {args.quant} -> {out_gguf}...")
        subprocess.run(
            ["llama-quantize.exe", str(f16_gguf), str(out_gguf), args.quant],
            check=True,
        )

    size_mb = out_gguf.stat().st_size / (1024 * 1024)
    print(f"GGUF gerado com sucesso: {out_gguf} ({size_mb:.1f} MiB)")
    shutil.rmtree(merged_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
