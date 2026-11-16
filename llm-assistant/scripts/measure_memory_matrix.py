"""Mede a matriz 2x2 de uso de memória (Tamanho 1.7B/0.6B x Original/Quantizado)."""

from __future__ import annotations

import gc
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx
import psutil
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common import DIR_EVAL


def measure_bf16_vram(base_model: str, adapter_dir: Path) -> float:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "Árvore com galho encostado em fio elétrico, quem acionar?"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tok(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    peak_mib = torch.cuda.max_memory_allocated() / (1024**2)
    del model, inputs
    gc.collect()
    torch.cuda.empty_cache()
    return round(peak_mib, 1)


def measure_gguf_pure_cpu_rss(gguf_path: Path, port: int = 8091) -> dict[str, float]:
    proc = subprocess.Popen(
        [
            "llama-server.exe",
            "-m",
            str(gguf_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "-dev",
            "none",
            "-ngl",
            "0",
            "-c",
            "2048",
            "-np",
            "1",
            "-t",
            "2",
            "--no-webui",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    ps_proc = psutil.Process(proc.pid)
    peak_rss = 0.0
    try:
        for _ in range(120):
            try:
                if httpx.get(f"http://127.0.0.1:{port}/health", timeout=1.0).status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(0.25)
        peak_rss = max(peak_rss, ps_proc.memory_info().rss / (1024**2))
        httpx.post(
            f"http://127.0.0.1:{port}/completion",
            json={"prompt": "Árvore com galho no fio elétrico?", "n_predict": 32, "temperature": 0.0},
            timeout=60.0,
        )
        peak_rss = max(peak_rss, ps_proc.memory_info().rss / (1024**2))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    return {
        "cpu_rss_mib": round(peak_rss, 1),
        "disk_mib": round(gguf_path.stat().st_size / (1024**2), 1),
    }


def main() -> None:
    rss_17b_f16 = measure_gguf_pure_cpu_rss(ROOT / "models" / "arborgrid-qwen3-1.7b-f16.gguf", port=8093)
    rss_06b_f16 = measure_gguf_pure_cpu_rss(ROOT / "models" / "arborgrid-qwen3-0.6b-f16.gguf", port=8094)
    payload = {
        "1.7B": {
            "original_bf16_vram_mib": measure_bf16_vram("Qwen/Qwen3-1.7B", ROOT / "adapters" / "lora_arborgrid_r16"),
            **measure_gguf_pure_cpu_rss(ROOT / "models" / "arborgrid-qwen3-1.7b-q4_k_m.gguf", port=8091),
            "f16_cpu_rss_mib": rss_17b_f16["cpu_rss_mib"],
            "f16_disk_mib": rss_17b_f16["disk_mib"],
        },
        "0.6B": {
            "original_bf16_vram_mib": measure_bf16_vram("Qwen/Qwen3-0.6B", ROOT / "adapters" / "lora_arborgrid_06b_r16"),
            **measure_gguf_pure_cpu_rss(ROOT / "models" / "arborgrid-qwen3-0.6b-q4_k_m.gguf", port=8092),
            "f16_cpu_rss_mib": rss_06b_f16["cpu_rss_mib"],
            "f16_disk_mib": rss_06b_f16["disk_mib"],
        },
    }
    out_path = DIR_EVAL / "matriz_memoria_2x2.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Matriz salva em {out_path}")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
