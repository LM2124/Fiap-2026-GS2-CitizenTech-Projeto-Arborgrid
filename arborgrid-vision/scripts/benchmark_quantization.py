"""
ArborGrid Vision — Benchmark de Quantização do SegFormer-B2 V3 Ouro (1024px)

Compara 5 configurações sobre a partição congelada de validação (8 imagens / 4 árvores)
e o teste cego sem fiação da Av. Paulista (20 imagens):
  1. pytorch_fp32         (PyTorch FP32 baseline local)
  2. onnx_fp32            (ONNX Runtime FP32 — controle de runtime)
  3. onnx_fp16            (ONNX FP16 via onnxconverter_common)
  4. onnx_int8_dynamic    (ONNX Runtime Quantização Dinâmica INT8)
  5. onnx_int8_static_qdq (ONNX Runtime Quantização Estática QDQ INT8 calibrada)

Cada variante roda em subprocesso isolado (--worker) em CPU (2 threads) para medir
pico de RAM RSS real, latência média/P95, tamanho em disco e gzip web, mIoU,
IoU de fiação, falsos positivos na Paulista e concordância de triagem morfológica.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnxconverter_common import float16
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

CV_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CV_ROOT))

from scripts.run_v3_inference_and_update_map import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    THRESHOLDS,
    V3_WEIGHTS,
    analyze_conflict_morphology_calibrated,
    load_v3_model,
)

IMGSZ = 1024
CLASS_MAP = {0: "copa", 1: "fiacao", 2: "poste"}
VAL_TREE_IDS = {"597616", "597652", "597668", "597842"}

ZIP_DATASET = CV_ROOT / "arborgrid_v3_gold_dataset.zip"
OUT_DIR = CV_ROOT / "runs" / "quantization_benchmark"
CACHE_DIR = OUT_DIR / "eval_cache"
MODELS_OUT = OUT_DIR / "models"


class _SegformerLogitsWrapper(nn.Module):
    def __init__(self, segformer_model: nn.Module) -> None:
        super().__init__()
        self.model = segformer_model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=pixel_values).logits


class _ZipCalibrationReader(CalibrationDataReader):
    def __init__(self, image_paths: list[Path], input_name: str) -> None:
        self._paths = image_paths
        self._input_name = input_name
        self._idx = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._idx >= len(self._paths):
            return None
        path = self._paths[self._idx]
        self._idx += 1
        x = preprocess_image(path)
        return {self._input_name: x}

    def rewind(self) -> None:
        self._idx = 0


def ensure_min_wire_thickness(pts: list, w: int, h: int, min_thick_px: float = 4.5) -> list:
    arr = np.array(pts, dtype=np.float32)
    area = cv2.contourArea(arr)
    peri = cv2.arcLength(arr, True)
    thick = 2.0 * area / max(peri, 1.0)
    if thick >= min_thick_px:
        return pts
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(arr).astype(np.int32)], 255)
    cv2.polylines(mask, [np.round(arr).astype(np.int32)], isClosed=True, color=255, thickness=5)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return pts
    biggest = max(cnts, key=cv2.contourArea)
    approx = cv2.approxPolyDP(biggest, 0.003 * cv2.arcLength(biggest, True), True)
    if len(approx) < 3:
        return pts
    return [[float(p[0][0]), float(p[0][1])] for p in approx]


def json_to_class_masks(json_path: Path, h: int = IMGSZ, w: int = IMGSZ) -> dict[int, np.ndarray]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    orig_h = float(data.get("imageHeight") or h)
    orig_w = float(data.get("imageWidth") or w)
    scale_x = w / max(orig_w, 1.0)
    scale_y = h / max(orig_h, 1.0)

    masks = {
        0: np.zeros((h, w), dtype=np.uint8),
        1: np.zeros((h, w), dtype=np.uint8),
        2: np.zeros((h, w), dtype=np.uint8),
    }

    for shape in data.get("shapes") or []:
        lbl = str(shape.get("label") or "").strip().lower()
        cid = 0 if "copa" in lbl else 1 if ("fio" in lbl or "fiacao" in lbl) else 2 if "poste" in lbl else None
        if cid is None:
            continue
        raw_pts = shape.get("points") or []
        if len(raw_pts) >= 3 and shape.get("shape_type", "polygon") != "line":
            pts = [[p[0] * scale_x, p[1] * scale_y] for p in raw_pts]
            if cid == 1:
                pts = ensure_min_wire_thickness(pts, w, h, min_thick_px=4.5)
            cv2.fillPoly(masks[cid], [np.round(np.array(pts)).astype(np.int32)], 1)
        elif len(raw_pts) >= 2:
            scaled_pts = np.round(np.array([[p[0] * scale_x, p[1] * scale_y] for p in raw_pts])).astype(np.int32)
            cv2.polylines(masks[cid], [scaled_pts], False, 1, thickness=max(3, int(5 * scale_x)))

    return masks


def compute_global_iou(pred_list: list[dict[int, np.ndarray]], gt_list: list[dict[int, np.ndarray]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    ious: list[float] = []
    for cid, cname in CLASS_MAP.items():
        tp = fp = fn = 0
        for pm, gm in zip(pred_list, gt_list):
            p = pm[cid] > 0
            g = gm[cid] > 0
            tp += int(np.count_nonzero(p & g))
            fp += int(np.count_nonzero(p & ~g))
            fn += int(np.count_nonzero(~p & g))
        iou = 100.0 * tp / max(tp + fp + fn, 1)
        precision = 100.0 * tp / max(tp + fp, 1)
        recall = 100.0 * tp / max(tp + fn, 1)
        summary[f"{cname}_iou"] = round(float(iou), 4)
        summary[f"{cname}_precision"] = round(float(precision), 4)
        summary[f"{cname}_recall"] = round(float(recall), 4)
        ious.append(iou)
    summary["miou"] = round(float(np.mean(ious)), 4)
    return summary


def preprocess_image(img_path: Path) -> np.ndarray:
    img_bgr = cv2.imread(str(img_path))
    if img_bgr.shape[:2] != (IMGSZ, IMGSZ):
        img_bgr = cv2.resize(img_bgr, (IMGSZ, IMGSZ), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_f = (img_rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(img_f, (2, 0, 1))[None, ...].astype(np.float32)


def logits_to_probs(logits: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(logits).float()
    probs = torch.sigmoid(
        F.interpolate(t, size=(IMGSZ, IMGSZ), mode="bilinear", align_corners=False)
    )[0].numpy()
    return probs


def prepare_eval_cache() -> dict[str, Any]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    manifest_file = CACHE_DIR / "partitions.json"
    if manifest_file.exists():
        return json.loads(manifest_file.read_text(encoding="utf-8"))

    with zipfile.ZipFile(ZIP_DATASET, "r") as zf:
        zf.extractall(CACHE_DIR)

    p1_dir = CACHE_DIR / "pacote_1"
    p2_dir = CACHE_DIR / "pacote_2"
    p3_dir = CACHE_DIR / "pacote_3_hard_negatives"
    blind_dir = CACHE_DIR / "test_paulista_blind"

    val_items: list[dict[str, str]] = []
    train_images: list[str] = []

    for jp in sorted(p1_dir.glob("*.json")):
        ip = jp.with_suffix(".jpg")
        if not ip.exists():
            continue
        if any(vid in jp.stem for vid in VAL_TREE_IDS):
            val_items.append({"img": str(ip), "json": str(jp), "stem": ip.stem})
        else:
            train_images.append(str(ip))

    for d in (p2_dir, p3_dir):
        for jp in sorted(d.glob("*.json")):
            ip = jp.with_suffix(".jpg")
            if ip.exists():
                train_images.append(str(ip))

    # Amostra estratificada de 24 imagens disjuntas da validação para calibração estática INT8
    step = max(len(train_images) // 24, 1)
    calib_images = train_images[::step][:24]

    blind_items = [
        {"img": str(ip), "stem": ip.stem}
        for ip in sorted(blind_dir.glob("*.jpg"))
    ]

    info = {
        "val_items": val_items,
        "blind_items": blind_items,
        "calib_images": calib_images,
    }
    manifest_file.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def measure_gzip_mib(file_path: Path) -> float:
    gz_path = file_path.with_suffix(file_path.suffix + ".gz")
    if not gz_path.exists():
        with file_path.open("rb") as f_in, gzip.open(gz_path, "wb", compresslevel=6) as f_out:
            shutil.copyfileobj(f_in, f_out)
    return round(gz_path.stat().st_size / (1024 * 1024), 2)


def export_and_quantize_models(calib_images: list[Path]) -> dict[str, Path]:
    MODELS_OUT.mkdir(parents=True, exist_ok=True)
    p_fp32 = MODELS_OUT / "segformer_v3_fp32.onnx"
    p_pre = MODELS_OUT / "segformer_v3_pre.onnx"
    p_fp16 = MODELS_OUT / "segformer_v3_fp16.onnx"
    p_dyn_mm = MODELS_OUT / "segformer_v3_int8_dyn_matmul.onnx"
    p_dyn = MODELS_OUT / "segformer_v3_int8_dynamic.onnx"
    p_stat = MODELS_OUT / "segformer_v3_int8_static_qdq.onnx"

    if not p_fp32.exists():
        print("[1/4] Exportando SegFormer-B2 V3 para ONNX FP32...")
        model = load_v3_model("cpu")
        wrapper = _SegformerLogitsWrapper(model).eval()
        dummy = torch.randn(1, 3, IMGSZ, IMGSZ)
        torch.onnx.export(
            wrapper,
            dummy,
            str(p_fp32),
            input_names=["pixel_values"],
            output_names=["logits"],
            opset_version=17,
            do_constant_folding=True,
        )

    if not p_pre.exists():
        print("[2/4] Pré-processando grafo ONNX para quantização (shape inference + folding)...")
        quant_pre_process(str(p_fp32), str(p_pre), skip_optimization=False)

    if not p_fp16.exists():
        print("[3/4] Convertendo ONNX FP32 -> FP16 (keep_io_types=True)...")
        m_fp32 = onnx.load(str(p_fp32))
        m_fp16 = float16.convert_float_to_float16(m_fp32, keep_io_types=True)
        onnx.save(m_fp16, str(p_fp16))

    if not p_dyn_mm.exists():
        print("[4/4a] Aplicando Quantização Dinâmica INT8 Seletiva (MatMul/Gemm)...")
        quantize_dynamic(
            model_input=str(p_pre),
            model_output=str(p_dyn_mm),
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul", "Gemm"],
        )

    if not p_dyn.exists():
        print("[4/4b] Aplicando Quantização Dinâmica INT8 Completa (Conv+MatMul QInt8)...")
        quantize_dynamic(
            model_input=str(p_pre),
            model_output=str(p_dyn),
            weight_type=QuantType.QInt8,
        )

    if not p_stat.exists():
        print(f"[4/4c] Aplicando Quantização Estática QDQ INT8 (calibração com {len(calib_images)} imagens)...")
        reader = _ZipCalibrationReader(calib_images, "pixel_values")
        quantize_static(
            model_input=str(p_pre),
            model_output=str(p_stat),
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            calibrate_method=CalibrationMethod.MinMax,
        )

    return {
        "pytorch_fp32": V3_WEIGHTS,
        "onnx_fp32": p_fp32,
        "onnx_fp16": p_fp16,
        "onnx_int8_dyn_matmul": p_dyn_mm,
        "onnx_int8_dynamic": p_dyn,
        "onnx_int8_static_qdq": p_stat,
    }


def _make_ort_session(onnx_path: Path, threads: int = 2) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])


def run_worker(variant: str, model_path: Path, partitions: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(2)
    proc = psutil.Process()
    peak_rss_mib = proc.memory_info().rss / (1024**2)

    val_items = partitions["val_items"]
    blind_items = partitions["blind_items"]

    if variant == "pytorch_fp32":
        model = load_v3_model("cpu")
        def _infer(x_np: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                xt = torch.from_numpy(x_np)
                return model(pixel_values=xt).logits.numpy()
    else:
        sess = _make_ort_session(model_path, threads=2)
        in_name = sess.get_inputs()[0].name
        def _infer(x_np: np.ndarray) -> np.ndarray:
            return sess.run(None, {in_name: x_np})[0]

    # Warmup para estabilizar alocadores/cache em CPU
    warm_x = preprocess_image(Path(val_items[0]["img"]))
    _ = _infer(warm_x)
    peak_rss_mib = max(peak_rss_mib, proc.memory_info().rss / (1024**2))

    latencies_ms: list[float] = []
    pred_list: list[dict[int, np.ndarray]] = []
    gt_list: list[dict[int, np.ndarray]] = []
    triages: dict[str, dict[str, Any]] = {}

    for item in val_items:
        x_np = preprocess_image(Path(item["img"]))
        t0 = time.perf_counter()
        logits = _infer(x_np)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt_ms)
        peak_rss_mib = max(peak_rss_mib, proc.memory_info().rss / (1024**2))

        probs = logits_to_probs(logits)
        gt_masks = json_to_class_masks(Path(item["json"]), IMGSZ, IMGSZ)
        pred_masks = {cid: (probs[cid] >= THRESHOLDS[cid]).astype(np.uint8) for cid in CLASS_MAP}
        pred_list.append(pred_masks)
        gt_list.append(gt_masks)

        triage, px_dir, px_wind, wire_px, copa_px, *_ = analyze_conflict_morphology_calibrated(
            pred_masks[0], pred_masks[1]
        )
        triages[item["stem"]] = {
            "triage": triage,
            "px_direct": px_dir,
            "px_wind": px_wind,
            "wire_px": wire_px,
            "copa_px": copa_px,
        }

    val_metrics = compute_global_iou(pred_list, gt_list)

    wire_fp_pixels: list[int] = []
    clean_images_count = 0
    for item in blind_items:
        x_np = preprocess_image(Path(item["img"]))
        t0 = time.perf_counter()
        logits = _infer(x_np)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt_ms)
        peak_rss_mib = max(peak_rss_mib, proc.memory_info().rss / (1024**2))

        probs = logits_to_probs(logits)
        pred_masks = {cid: (probs[cid] >= THRESHOLDS[cid]).astype(np.uint8) for cid in CLASS_MAP}

        wire_pred = pred_masks[1]
        cnts, _ = cv2.findContours(wire_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clean_pred = np.zeros_like(wire_pred)
        for c in cnts:
            if cv2.contourArea(c) >= 25:
                cv2.drawContours(clean_pred, [c], -1, 1, -1)
        fp_px = int(np.count_nonzero(clean_pred))
        wire_fp_pixels.append(fp_px)
        if fp_px == 0:
            clean_images_count += 1

        triage, px_dir, px_wind, wire_px, copa_px, *_ = analyze_conflict_morphology_calibrated(
            pred_masks[0], pred_masks[1]
        )
        triages[item["stem"]] = {
            "triage": triage,
            "px_direct": px_dir,
            "px_wind": px_wind,
            "wire_px": wire_px,
            "copa_px": copa_px,
        }

    raw_mib = round(model_path.stat().st_size / (1024 * 1024), 2)
    gz_mib = measure_gzip_mib(model_path)

    result = {
        "variant": variant,
        "model_file": model_path.name,
        "size_mib": raw_mib,
        "gzip_mib": gz_mib,
        "peak_ram_rss_mib": round(peak_rss_mib, 1),
        "cpu_threads": 2,
        "n_images_timed": len(latencies_ms),
        "latency_mean_ms": round(float(np.mean(latencies_ms)), 1),
        "latency_p95_ms": round(float(np.percentile(latencies_ms, 95)), 1),
        "val_metrics": val_metrics,
        "paulista_blind": {
            "mean_fp_pixels": round(float(np.mean(wire_fp_pixels)), 2),
            "zero_fp_rate": round(100.0 * clean_images_count / max(len(wire_fp_pixels), 1), 2),
            "clean_images": clean_images_count,
            "total_images": len(wire_fp_pixels),
        },
        "triages": triages,
    }
    return result


def _plot_quantization_summary(all_results: list[dict[str, Any]], out_png: Path) -> None:
    """Gera painel comparativo 2x2 de tamanho, qualidade, latência e memória."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    label_map = {
        "pytorch_fp32": "PyTorch\nFP32",
        "onnx_fp32": "ONNX\nFP32",
        "onnx_fp16": "ONNX\nFP16",
        "onnx_int8_dyn_matmul": "ONNX INT8\nMatMul+Gemm",
        "onnx_int8_dynamic": "ONNX INT8\nDinâmico Full",
        "onnx_int8_static_qdq": "ONNX INT8\nEstático QDQ",
    }
    labels = [label_map.get(r["variant"], r["variant"]) for r in all_results]
    x = np.arange(len(all_results))

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.8), dpi=160)
    fig.patch.set_facecolor("#ffffff")
    fig.suptitle(
        "ArborGrid Vision — Quantização e Exportação SegFormer-B2 V3 (1024×1024, CPU 2 Threads)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # 1) Tamanho em disco vs Gzip (MiB)
    ax = axes[0, 0]
    raw_sizes = [r["size_mib"] for r in all_results]
    gz_sizes = [r["gzip_mib"] for r in all_results]
    w = 0.36
    b1 = ax.bar(x - w / 2, raw_sizes, w, label="Peso Bruto (MiB)", color="#3b82f6")
    b2 = ax.bar(x + w / 2, gz_sizes, w, label="Comprimido Gzip (MiB)", color="#0f766e")
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1.8, f"{h:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("1. Tamanho do Artefato (MiB)", fontweight="bold", pad=10)
    ax.set_ylabel("MiB")
    ax.set_ylim(0, max(raw_sizes) * 1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper right", fontsize=8.5)

    # 2) Qualidade e Robustez (%)
    ax = axes[0, 1]
    mious = [r["val_metrics"]["miou"] for r in all_results]
    fios = [r["val_metrics"]["fiacao_iou"] for r in all_results]
    zfps = [r["paulista_blind"]["zero_fp_rate"] for r in all_results]
    triages = [r["triage_agreement_pct"] for r in all_results]
    w4 = 0.20
    ax.bar(x - 1.5 * w4, mious, w4, label="mIoU Val (%)", color="#2563eb")
    ax.bar(x - 0.5 * w4, fios, w4, label="IoU Fiação (%)", color="#d97706")
    ax.bar(x + 0.5 * w4, zfps, w4, label="Zero-FP Paulista (%)", color="#16a34a")
    ax.bar(x + 1.5 * w4, triages, w4, label="Concord. Triagem (%)", color="#7c3aed")
    for idx in range(len(all_results)):
        ax.text(x[idx] + 0.5 * w4, zfps[idx] + 1.5, f"{zfps[idx]:.0f}%", ha="center", va="bottom", fontsize=7.5, color="#15803d")
        ax.text(x[idx] - 0.5 * w4, fios[idx] + 1.5, f"{fios[idx]:.1f}", ha="center", va="bottom", fontsize=7.5, color="#b45309")
    ax.set_title("2. Qualidade de Segmentação, Zero-FP e Triagem (%)", fontweight="bold", pad=10)
    ax.set_ylabel("%")
    ax.set_ylim(0, 118)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper right", ncol=2, fontsize=8)

    # 3) Latência média por imagem (s) em CPU 2 threads
    ax = axes[1, 0]
    lat_s = [r["latency_mean_ms"] / 1000.0 for r in all_results]
    colors_lat = ["#64748b", "#0284c7", "#0ea5e9", "#16a34a", "#dc2626", "#f59e0b"]
    bars_lat = ax.bar(x, lat_s, 0.55, color=colors_lat)
    for bar, r in zip(bars_lat, all_results):
        h = bar.get_height()
        p95 = r["latency_p95_ms"] / 1000.0
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.6, f"{h:.2f}s\n(P95 {p95:.2f}s)", ha="center", va="bottom", fontsize=8)
    ax.set_title("3. Latência de Inferência por Imagem (CPU 2 Threads)", fontweight="bold", pad=10)
    ax.set_ylabel("Segundos (menor é melhor)")
    ax.set_ylim(0, max(lat_s) * 1.24)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # 4) Pico de RAM RSS (GiB)
    ax = axes[1, 1]
    rss_gib = [r["peak_ram_rss_mib"] / 1024.0 for r in all_results]
    bars_ram = ax.bar(x, rss_gib, 0.55, color="#475569")
    for bar, r in zip(bars_ram, all_results):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.14, f"{h:.2f} GiB\n({r['peak_ram_rss_mib']:.0f} MiB)", ha="center", va="bottom", fontsize=8)
    ax.set_title("4. Pico de Memória RAM RSS do Processo (GiB)", fontweight="bold", pad=10)
    ax.set_ylabel("GiB RSS (menor é melhor)")
    ax.set_ylim(0, max(rss_gib) * 1.25)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout(rect=[0, 0.01, 1, 0.94])
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    partitions = prepare_eval_cache()

    if args.worker:
        res = run_worker(args.worker, Path(args.model_path), partitions)
        worker_out = OUT_DIR / f"worker_{args.worker}.json"
        worker_out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[{args.worker}] concluído -> {worker_out}")
        return

    calib_paths = [Path(p) for p in partitions["calib_images"]]
    model_paths = export_and_quantize_models(calib_paths)

    variants_order = [
        "pytorch_fp32",
        "onnx_fp32",
        "onnx_fp16",
        "onnx_int8_dyn_matmul",
        "onnx_int8_dynamic",
        "onnx_int8_static_qdq",
    ]
    all_results: list[dict[str, Any]] = []
    for variant in variants_order:
        worker_json = OUT_DIR / f"worker_{variant}.json"
        if args.summarize_only and worker_json.exists():
            print(f"=== Reaproveitando resultado em cache: {variant} ({worker_json.name}) ===")
        else:
            print(f"\n=== Benchmark isolado (CPU 2t): {variant} ({model_paths[variant].name}) ===")
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                variant,
                "--model-path",
                str(model_paths[variant]),
            ]
            subprocess.run(cmd, check=True)
        all_results.append(json.loads(worker_json.read_text(encoding="utf-8")))

    ref_triages = all_results[0]["triages"]
    total_stems = len(ref_triages)
    for row in all_results:
        matches = sum(
            1 for stem, ref in ref_triages.items()
            if row["triages"].get(stem, {}).get("triage") == ref["triage"]
        )
        row["triage_agreement_pct"] = round(100.0 * matches / max(total_stems, 1), 2)
        row["triage_matches"] = f"{matches}/{total_stems}"

    report_path = OUT_DIR / "quantization_benchmark_report.json"
    report_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_path = OUT_DIR / "quantizacao_segformer_b2_v3.png"
    _plot_quantization_summary(all_results, plot_path)
    print(f"\nRelatório consolidado salvo em: {report_path}")
    print(f"Gráfico salvo em: {plot_path}\n")

    header = (
        f"{'Variante':<22} | {'Disco':>9} | {'Gzip Web':>9} | {'RAM RSS':>9} | "
        f"{'Latência':>9} | {'mIoU':>7} | {'IoU Fio':>7} | {'ZeroFP':>7} | {'Triagem':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in all_results:
        vm = r["val_metrics"]
        pb = r["paulista_blind"]
        print(
            f"{r['variant']:<22} | {r['size_mib']:>5.1f} MiB | {r['gzip_mib']:>5.1f} MiB | "
            f"{r['peak_ram_rss_mib']:>5.0f} MiB | {r['latency_mean_ms']/1000.0:>7.2f}s | "
            f"{vm['miou']:>6.2f}% | {vm['fiacao_iou']:>6.2f}% | "
            f"{pb['zero_fp_rate']:>6.1f}% | {r['triage_agreement_pct']:>6.1f}%"
        )


if __name__ == "__main__":
    main()
