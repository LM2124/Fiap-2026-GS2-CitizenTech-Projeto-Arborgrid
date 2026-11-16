"""Avalia um modelo GGUF quantizado em CPU pura via llama-server.exe (-ngl 0).

Simula o perfil de vCPU da EC2 AWS (sem GPU), medindo RSS RAM, tamanho em disco,
latência média/p95 e todas as métricas programáticas em avaliacao/test_set.jsonl.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import psutil
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common import (
    BASE_MODEL,
    DEFAULT_RETRIEVAL_THRESHOLD,
    DIR_EVAL,
    format_chat_prompt,
    load_jsonl,
)
from src.retrieval import KnowledgeRetriever
from src.tools import ArborGridInspector
from src.validator import apply_post_generation_filters, evaluate_prediction


def _wait_ready(base_url: str, timeout_s: float = 60.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            resp = httpx.get(f"{base_url}/health", timeout=2.0)
            if resp.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"llama-server não ficou pronto em {timeout_s}s em {base_url}")


def _summarize(rows: list[dict[str, Any]], label: str, disk_mb: float, peak_rss_mb: float) -> dict[str, Any]:
    n = len(rows)
    oos_rows = [r for r in rows if r["categoria"] == "nao_consta"]
    ans_rows = [r for r in rows if r["categoria"] != "nao_consta"]

    def _rate(subset: list[dict[str, Any]], key: str) -> float:
        return round(sum(1 for r in subset if r[key]) / max(len(subset), 1), 4)

    latencies = [r["latency_ms"] for r in rows]
    total_tokens = sum(r["generated_tokens"] for r in rows)
    total_secs = sum(latencies) / 1000.0
    decode_tps_vals = [r["tps_decode"] for r in rows if r.get("tps_decode")]
    return {
        "configuracao": label,
        "total_casos": n,
        "tamanho_gguf_mib": round(disk_mb, 1),
        "pico_ram_rss_mib": round(peak_rss_mb, 1),
        "json_valido": _rate(rows, "json_valid"),
        "schema_valido": _rate(rows, "schema_valid"),
        "acuracia_risco": _rate(rows, "risk_exact"),
        "acuracia_canal": _rate(rows, "channel_exact"),
        "seguranca_poda": _rate(rows, "prune_exact"),
        "citacao_fonte": _rate(ans_rows, "source_cited"),
        "recusa_nao_consta": _rate(oos_rows, "refusal_exact"),
        "alucinacao_fonte": _rate(rows, "hallucinated_source"),
        "alucinacao_tel": _rate(rows, "hallucinated_phone"),
        "lgpd_limpo": _rate(rows, "lgpd_clean"),
        "acuracia_geral_estrita": _rate(rows, "overall_exact"),
        "latencia_media_ms": round(float(np.mean(latencies)), 1),
        "latencia_p95_ms": round(float(np.percentile(latencies, 95)), 1),
        "tokens_medios": round(total_tokens / max(n, 1), 1),
        "tps_medio": round(total_tokens / max(total_secs, 1e-6), 2),
        "tps_decode_medio": round(float(np.mean(decode_tps_vals)), 2) if decode_tps_vals else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--tokenizer-dir", default=BASE_MODEL)
    parser.add_argument("--label", required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--gpu-layers", type=int, default=0)
    parser.add_argument("--port", type=int, default=8089)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    gguf_path = Path(args.gguf)
    disk_mb = gguf_path.stat().st_size / (1024 * 1024)
    base_url = f"http://127.0.0.1:{args.port}"

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    retriever = KnowledgeRetriever(device="cpu")
    inspector = ArborGridInspector()
    test_cases = load_jsonl(DIR_EVAL / "test_set.jsonl")

    server_cmd = [
        "llama-server.exe",
        "-m",
        str(gguf_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
    ]
    if args.gpu_layers == 0:
        server_cmd.extend(["-dev", "none", "-ngl", "0"])
    else:
        server_cmd.extend(["-ngl", str(args.gpu_layers)])
    server_cmd.extend(["-c", "2048", "-np", "1", "-t", str(args.threads), "--no-webui"])

    # Argumentos de lista explicitos (sem shell=True) para benchmark local controlado
    proc = subprocess.Popen(server_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # nosemgrep
    ps_proc = psutil.Process(proc.pid)
    peak_rss_mb = 0.0

    try:
        _wait_ready(base_url, timeout_s=60.0)
        peak_rss_mb = max(peak_rss_mb, ps_proc.memory_info().rss / (1024 * 1024))
        results: list[dict[str, Any]] = []

        for case in test_cases:
            chunks = retriever.retrieve(
                case["pergunta"],
                top_k=3,
                min_score=DEFAULT_RETRIEVAL_THRESHOLD,
            )
            tree_ctx, _ = inspector.maybe_build_tool_context(case["pergunta"])
            if tree_ctx is not None:
                present_ids = {c["chunk_id"] for c in chunks}
                for required_id in ("ARBORGRID-TRIAGEM", "SP156-ROTINA"):
                    if required_id not in present_ids:
                        match = next((dict(c) for c in retriever.chunks if c["chunk_id"] == required_id), None)
                        if match is not None:
                            match["score"] = 0.70
                            chunks.append(match)

            prompt = format_chat_prompt(
                tokenizer=tokenizer,
                question=case["pergunta"],
                chunks=chunks,
                tree_context=tree_ctx,
            )
            t0 = time.perf_counter()
            resp = httpx.post(
                f"{base_url}/completion",
                json={
                    "prompt": prompt,
                    "n_predict": 240,
                    "temperature": 0.0,
                    "top_k": 1,
                    "top_p": 1.0,
                    "stop": ["<|im_end|>", "<|endoftext|>"],
                },
                timeout=120.0,
            )
            resp.raise_for_status()
            lat_ms = (time.perf_counter() - t0) * 1000.0
            payload = resp.json()
            raw_out = payload.get("content", "").strip()
            n_tokens = int(payload.get("tokens_predicted") or len(tokenizer.encode(raw_out, add_special_tokens=False)))
            decode_tps = payload.get("timings", {}).get("predicted_per_second")
            peak_rss_mb = max(peak_rss_mb, ps_proc.memory_info().rss / (1024 * 1024))

            filtered_out = apply_post_generation_filters(raw_out, case["pergunta"], chunks)
            metrics = evaluate_prediction(
                case=case,
                raw_output=filtered_out,
                retrieved_chunks=chunks,
                tree_context=tree_ctx,
            )
            metrics["latency_ms"] = round(lat_ms, 1)
            metrics["generated_tokens"] = n_tokens
            metrics["tps"] = round(n_tokens / max(lat_ms / 1000.0, 1e-6), 2)
            metrics["tps_decode"] = round(float(decode_tps), 2) if decode_tps else None
            results.append(metrics)

        summary = _summarize(results, args.label, disk_mb, peak_rss_mb)
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({"summary": summary, "details": results}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print("=== RESULTADO GGUF CPU ===")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
