"""
ArborGrid - Utilitário de Inspeção e Auditoria Rápida de Rótulos

Uso no terminal:
  python scripts/inspect_label_audit.py tree_612770_center_front_h36
  python scripts/inspect_label_audit.py 612770_center
  python scripts/inspect_label_audit.py 612770_center --crop 380 0 80 640
  python scripts/inspect_label_audit.py 612770_center --shape 18
"""

import sys
import json
import argparse
import cv2
import numpy as np
from pathlib import Path

CV_ROOT = Path(__file__).resolve().parent.parent
WS_DIR = CV_ROOT / "dataset" / "workspace_xanylabeling"
PACK_BASE = WS_DIR / "pacote_colega_1"
PACK_LORENZO = WS_DIR / "pacote_colega_1_labelled"
PACK_REVISED = WS_DIR / "pacote_colega_1_revised"
ANOM_DIR = CV_ROOT / "runs" / "pilot_ablation" / "anomaly_candidates"

COLORS_BGR = {
    "copa": (60, 195, 45),     # Verde
    "fiacao": (35, 45, 245),   # Vermelho
    "poste": (235, 165, 20),   # Amarelo/Dourado
}


def find_stem(query: str) -> str:
    query = query.strip().replace(".jpg", "").replace(".json", "")
    all_stems = [p.stem for p in PACK_REVISED.glob("*.jpg")]
    for s in all_stems:
        if s == query:
            return s
    for s in all_stems:
        if query in s:
            return s
    raise FileNotFoundError(f"Nenhuma imagem encontrada com a busca '{query}'.")


def load_shapes(json_path: Path):
    if not json_path.exists():
        return []
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        return data.get("shapes", [])
    except Exception:
        return []


def render_panel(img_bgr, shapes, title, draw_shape_ids=False):
    canvas = img_bgr.copy()
    overlay = img_bgr.copy()
    for s in shapes:
        lbl = str(s.get("label", "")).strip().lower()
        color = COLORS_BGR.get(lbl, (200, 200, 200))
        pts = np.round(np.array(s.get("points", []))).astype(np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(overlay, [pts], color)

    cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, canvas)

    for i, s in enumerate(shapes):
        lbl = str(s.get("label", "")).strip().lower()
        color = COLORS_BGR.get(lbl, (200, 200, 200))
        pts = np.round(np.array(s.get("points", []))).astype(np.int32)
        if len(pts) >= 3:
            thickness = 2 if lbl != "copa" else 1
            cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)
            if draw_shape_ids:
                cx, cy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
                cv2.putText(canvas, str(i), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(canvas, str(i), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 26), (15, 23, 42), -1)
    cv2.putText(canvas, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (248, 250, 252), 1, cv2.LINE_AA)
    return canvas


def inspect(stem: str, crop_box=None, shape_idx=None, output_path=None):
    img_p = PACK_REVISED / f"{stem}.jpg"
    if not img_p.exists():
        raise FileNotFoundError(f"Imagem não encontrada: {img_p}")

    img = cv2.imread(str(img_p))
    h, w = img.shape[:2]

    base_shapes = load_shapes(PACK_BASE / f"{stem}.json")
    lor_shapes = load_shapes(PACK_LORENZO / f"{stem}.json")
    rev_shapes = load_shapes(PACK_REVISED / f"{stem}.json")

    # Estatísticas
    def count_by_cls(slist):
        from collections import Counter
        return Counter(s.get("label", "desconhecido") for s in slist)

    cb = count_by_cls(base_shapes)
    cl = count_by_cls(lor_shapes)
    cr = count_by_cls(rev_shapes)

    print("=" * 75)
    print(f"AUDITORIA: {stem}  ({w}x{h} px)")
    print("=" * 75)
    print(f"  * Pré-label GPU : {len(base_shapes):2d} shapes | {dict(cb)}")
    print(f"  * Lorenzo       : {len(lor_shapes):2d} shapes | {dict(cl)}")
    print(f"  * Sua Revisão   : {len(rev_shapes):2d} shapes | {dict(cr)}")

    # Lista shapes da revisão com bbox
    print("\n  Shapes na sua revisão:")
    for i, s in enumerate(rev_shapes):
        lbl = s.get("label", "")
        pts = np.array(s.get("points", []))
        if len(pts) > 0:
            x_min, y_min = np.min(pts, axis=0)
            x_max, y_max = np.max(pts, axis=0)
            bw, bh = int(x_max - x_min), int(y_max - y_min)
            print(f"    [{i:2d}] {lbl:7s} | bbox=({int(x_min):3d}, {int(y_min):3d}, w={bw:3d}, h={bh:3d}) | pts={len(pts):2d}")

    # Checa anomalia salva
    anom_p = ANOM_DIR / f"anom_{stem}.jpg"
    has_anom = anom_p.exists()
    if has_anom:
        print(f"\n  [!] Esta imagem possui diagnóstico no Scanner de Anomalias: {anom_p.name}")

    # Monta painéis
    p_orig = img.copy()
    cv2.rectangle(p_orig, (0, 0), (p_orig.shape[1], 26), (15, 23, 42), -1)
    cv2.putText(p_orig, "ORIGINAL", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (248, 250, 252), 1, cv2.LINE_AA)

    p_lor = render_panel(img, lor_shapes, f"LORENZO ({len(lor_shapes)} shapes)")
    p_rev = render_panel(img, rev_shapes, f"SUA REVISAO ({len(rev_shapes)} shapes)", draw_shape_ids=True)

    panels = [p_orig, p_lor, p_rev]

    if has_anom:
        anom_img = cv2.imread(str(anom_p))
        if anom_img is not None and anom_img.shape[1] >= w * 3:
            # Pega o 3º quadrante (predição da IA out-of-fold)
            oof_panel = anom_img[:, w * 2: w * 3]
            if oof_panel.shape[:2] == (h, w):
                panels.append(oof_panel)

    strip = np.hstack(panels)

    # Crop opcional
    if shape_idx is not None and 0 <= shape_idx < len(rev_shapes):
        pts = np.array(rev_shapes[shape_idx].get("points", []))
        if len(pts) > 0:
            x_min, y_min = np.min(pts, axis=0)
            x_max, y_max = np.max(pts, axis=0)
            margin = 35
            crop_box = [max(0, int(y_min - margin)), max(0, int(x_min - margin)),
                        min(h, int(y_max + margin)) - max(0, int(y_min - margin)),
                        min(w, int(x_max + margin)) - max(0, int(x_min - margin))]

    if crop_box:
        cy, cx, ch, cw = crop_box
        strip = strip[cy:cy + ch, :]
        print(f"\n  [Zoom aplicado] Crop vertical: y={cy}..{cy + ch}")

    if output_path is None:
        output_path = CV_ROOT / "runs" / "pilot_ablation" / f"audit_{stem}.jpg"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), strip, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    print(f"\n  -> Imagem de auditoria salva em: {output_path}")
    print("=" * 75)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Auditoria visual rápida de anotações ArborGrid")
    parser.add_argument("query", help="Nome do stem ou trecho do nome (ex: 612770_center)")
    parser.add_argument("--crop", nargs=4, type=int, metavar=("Y", "X", "H", "W"), help="Crop na região Y X H W")
    parser.add_argument("--shape", type=int, help="Dá zoom automático no shape N da sua revisão")
    parser.add_argument("--out", type=str, help="Caminho customizado de saída para o JPG")
    args = parser.parse_args()

    stem = find_stem(args.query)
    inspect(stem, crop_box=args.crop, shape_idx=args.shape, output_path=args.out)


if __name__ == "__main__":
    main()

