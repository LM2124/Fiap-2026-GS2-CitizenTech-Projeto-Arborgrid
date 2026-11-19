"""Regera os overlays do Mapa com o mesmo visual da Análise de Imagem do site.

Reproduz em Python o pipeline de `webapp/frontend/src/utils/visionOrt.ts` (o que o navegador faz):

1. SegFormer V3 (ONNX int8, o mesmo modelo servido ao navegador) em 1024x1024 com normalização ImageNet;
2. logits 256x256 -> 640x640 (bilinear, align_corners=False) e limiares V3
   (copa: sigmoide >= 0,50; fiação: sigmoide >= 0,40);
3. filtro de componentes (8-vizinhança): fiação >= 80 px; copa >= max(800 px, 15% do maior componente);
4. distância euclidiana à fiação: contato direto <= 6 px, margem de vento <= 24 px (só para a triagem de conferência);
5. mescla translúcida (padrão do site: 20%) com prioridade interseção > fiação > copa, sem a camada de vento,
   e contorno nível 2 (pincel 2x2) em copa, fiação e interseção.

Também calcula duas estatísticas legíveis por foto e grava em `views[].v3_analysis`:
- `contact_points`: trechos separados em que copa e fiação se sobrepõem (as manchas vermelhas); pedaços a
  até ~6 px um do outro contam como um trecho só, e cada trecho precisa de >= 25 px de sobreposição;
- `wire_contact_pct`: % dos pixels de fiação que estão dentro da copa.

Não altera triagens nem métricas existentes do manifesto (a triagem oficial continua a do pipeline V3);
apenas informa quantas vistas teriam triagem diferente neste reprocessamento.

Uso:
    python scripts/render_web_overlays.py --manifest ../webapp/data/manifest.json \
        --images dataset/sp_geosampa_api/images --out dataset/sp_geosampa_api/v3_overlays_web
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

IMGSZ = 1024
WORK_SZ = 640
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
LOGIT_THRESH_COPA = 0.0
LOGIT_THRESH_WIRE = math.log(0.4 / 0.6)

# Mesmas cores do site (RGB)
RGB_COPA = (45, 195, 60)
RGB_WIRE = (250, 204, 21)
RGB_INTERSECTION = (220, 38, 38)

DEFAULT_OPACITY = 0.20
MIN_CONTACT_AREA = 25
CONTACT_MERGE_RADIUS = 6
JPEG_QUALITY = 88


def filter_small_components(mask: np.ndarray, min_area: int, keep_dominant_ratio: float = 0.0) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(mask, dtype=bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    threshold = max(min_area, areas.max() * keep_dominant_ratio) if keep_dominant_ratio > 0 else min_area
    keep = np.zeros(n, dtype=bool)
    keep[1:] = areas >= threshold
    return keep[labels]


def border_of(mask: np.ndarray) -> np.ndarray:
    """Pixels da máscara com algum vizinho-4 fora dela (ou na borda da imagem), como no paintContour do site."""
    padded = np.pad(mask, 1, constant_values=False)
    interior = (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    return mask & ~interior


def brush_2x2(border: np.ndarray) -> np.ndarray:
    """Pincel do nível 2 do site: offsets (0,0), (1,0), (0,1), (1,1)."""
    out = border.copy()
    out[:, 1:] |= border[:, :-1]
    out[1:, :] |= border[:-1, :]
    out[1:, 1:] |= border[:-1, :-1]
    return out


def segment(session: ort.InferenceSession, rgb: np.ndarray) -> dict:
    resized = cv2.resize(rgb, (IMGSZ, IMGSZ), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    tensor = ((resized - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)[None].astype(np.float32)
    logits = session.run(None, {session.get_inputs()[0].name: tensor})[0][0]

    l_copa = cv2.resize(logits[0], (WORK_SZ, WORK_SZ), interpolation=cv2.INTER_LINEAR)
    l_wire = cv2.resize(logits[1], (WORK_SZ, WORK_SZ), interpolation=cv2.INTER_LINEAR)
    wire = filter_small_components(l_wire >= LOGIT_THRESH_WIRE, 80)
    copa = filter_small_components(l_copa >= LOGIT_THRESH_COPA, 800, 0.15)

    copa_px = int(copa.sum())
    wire_px = int(wire.sum())
    contact_direct_px = 0
    wind_buffer_px = 0
    if wire_px and copa_px:
        dist = cv2.distanceTransform((~wire).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        d2 = dist.astype(np.float64) ** 2
        contact_direct_px = int((copa & (d2 <= 36.0)).sum())
        wind_buffer_px = int((copa & (d2 > 36.0) & (d2 <= 576.0)).sum())

    triage = "SEGURO"
    if contact_direct_px >= 3000 and wire_px >= 800:
        triage = "CRITICO"
    elif (contact_direct_px >= 250 or wind_buffer_px >= 1200) and wire_px >= 200:
        triage = "PROXIMIDADE"

    intersection = copa & wire
    contact_points = 0
    if intersection.any():
        # Junta pedaços do mesmo fio separados por pequenas falhas da máscara antes de contar os trechos
        k = 2 * CONTACT_MERGE_RADIUS + 1
        merged = cv2.dilate(intersection.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        n, labels = cv2.connectedComponents(merged, connectivity=8)
        overlap_per_zone = np.bincount(labels[intersection], minlength=n)
        contact_points = int((overlap_per_zone[1:] >= MIN_CONTACT_AREA).sum())
    overlap_px = int(intersection.sum())
    wire_contact_pct = round(100.0 * overlap_px / wire_px, 1) if wire_px else 0.0

    return {
        "copa": copa,
        "wire": wire,
        "intersection": intersection,
        "copa_px": copa_px,
        "wire_px": wire_px,
        "contact_direct_px": contact_direct_px,
        "triage": triage,
        "contact_points": contact_points,
        "wire_contact_pct": wire_contact_pct,
    }


def render(rgb: np.ndarray, seg: dict, opacity: float) -> np.ndarray:
    base = cv2.resize(rgb, (WORK_SZ, WORK_SZ), interpolation=cv2.INTER_AREA) if rgb.shape[:2] != (WORK_SZ, WORK_SZ) else rgb
    out = base.astype(np.float32)
    copa, wire, inter = seg["copa"], seg["wire"], seg["intersection"]

    color = np.zeros_like(out)
    painted = np.zeros(copa.shape, dtype=bool)
    for mask, rgb_color in ((inter, RGB_INTERSECTION), (wire & ~inter, RGB_WIRE), (copa & ~wire, RGB_COPA)):
        color[mask] = rgb_color
        painted |= mask
    out[painted] = np.round(out[painted] * (1.0 - opacity) + color[painted] * opacity)

    for mask, rgb_color in ((copa, RGB_COPA), (wire, RGB_WIRE), (inter, RGB_INTERSECTION)):
        if mask.any():
            out[brush_2x2(border_of(mask))] = rgb_color
    return out.astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--manifest", type=Path, required=True, nargs="+", help="manifest(s) a atualizar (mesmo conteúdo)")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path(__file__).resolve().parents[1] / "models" / "segformer_v3_int8_dyn_matmul.onnx")
    parser.add_argument("--opacity", type=float, default=DEFAULT_OPACITY)
    args = parser.parse_args()

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = os.cpu_count() or 4
    session = ort.InferenceSession(str(args.model), sess_options=opts, providers=["CPUExecutionProvider"])
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.manifest[0].read_text(encoding="utf-8"))
    triage_cmp: Counter = Counter()
    contact_diffs: list[float] = []
    done = 0
    t0 = time.time()
    for tree in manifest["trees"]:
        for view in tree.get("views", []):
            v3 = view.get("v3_analysis") or {}
            ov_name = v3.get("v3_overlay_file")
            img_path = args.images / str(view.get("file_name", ""))
            if not ov_name or not img_path.is_file():
                continue
            bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            seg = segment(session, rgb)
            overlay = render(rgb, seg, args.opacity)
            cv2.imwrite(str(args.out / ov_name), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

            v3["contact_points"] = seg["contact_points"]
            v3["wire_contact_pct"] = seg["wire_contact_pct"]
            stored = str(v3.get("triage", ""))
            triage_cmp["igual" if stored == seg["triage"] else f"{stored}->{seg['triage']}"] += 1
            ref = int(v3.get("contact_direct_px", 0) or 0)
            if ref:
                contact_diffs.append(abs(seg["contact_direct_px"] - ref) / ref)
            done += 1
            if done % 50 == 0:
                print(f"{done} vistas ({time.time() - t0:.0f} s)", flush=True)

    manifest["overlays_render_version"] = datetime.now(timezone.utc).strftime("web-%Y%m%d%H%M")
    text = json.dumps(manifest, ensure_ascii=False, indent=2)
    for path in args.manifest:
        path.write_text(text, encoding="utf-8")

    print(f"\n{done} overlays em {args.out} ({time.time() - t0:.0f} s)")
    print("triagem (manifesto -> reprocessada):", dict(triage_cmp))
    if contact_diffs:
        print(f"contato direto: diferença relativa mediana {100 * float(np.median(contact_diffs)):.1f}% ({len(contact_diffs)} vistas com contato)")
    print("versão:", manifest["overlays_render_version"])


if __name__ == "__main__":
    main()
