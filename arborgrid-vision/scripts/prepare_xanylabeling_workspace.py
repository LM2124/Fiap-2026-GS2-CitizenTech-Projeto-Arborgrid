"""
ArborGrid - Preparador de Workspace Nativo para X-AnyLabeling
Módulo: arborgrid-vision / scripts / prepare_xanylabeling_workspace.py

Converte as imagens + pré-anotações YOLO-Seg geradas na GPU para o formato JSON nativo
do X-AnyLabeling (com simplificação suave de vértices para facilitar edição humana),
criando uma pasta pronta para abrir direto via `File -> Open Dir` (Ctrl + U).

Suporta também `--split-packs N` para dividir o trabalho entre N colegas do grupo
(agrupando sempre por árvore para manter contexto visual).
"""

import json
import math
import shutil
import argparse
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

CV_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = CV_ROOT / "dataset" / "sp_geosampa_api"
IMAGES_DIR = SRC_DIR / "images"
PRELABELS_DIR = SRC_DIR / "prelabels_yolo_seg"
DEFAULT_WORKSPACE = CV_ROOT / "dataset" / "workspace_xanylabeling"

CLASS_MAP = {
    0: "copa",
    1: "fiacao",
    2: "poste",
}


def yolo_seg_to_xanylabeling_shapes(txt_path: Path, width: int, height: int) -> list[dict]:
    """Lê .txt YOLO-Seg e converte para lista de shapes JSON do X-AnyLabeling com vértices enxutos."""
    if not txt_path.exists():
        return []

    shapes = []
    for raw_line in txt_path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.strip().split()
        if len(parts) < 7:  # cls + pelo menos 3 pares (x, y)
            continue
        cid = int(parts[0])
        label = CLASS_MAP.get(cid)
        if not label:
            continue

        coords = [float(v) for v in parts[1:]]
        pts = []
        for i in range(0, len(coords) - 1, 2):
            px = round(coords[i] * width, 2)
            py = round(coords[i + 1] * height, 2)
            pts.append([px, py])

        if len(pts) < 3:
            continue

        # Simplificar levemente polígonos muito densos para facilitar edição com mouse no X-AnyLabeling
        cnt = np.array(pts, dtype=np.float32).reshape((-1, 1, 2))
        peri = cv2.arcLength(cnt, True)
        eps_factor = 0.006 if label == "copa" else 0.010
        approx = cv2.approxPolyDP(cnt, eps_factor * peri, True)
        clean_pts = [[round(float(p[0][0]), 2), round(float(p[0][1]), 2)] for p in approx]
        if len(clean_pts) < 3:
            clean_pts = pts

        shapes.append(
            {
                "label": label,
                "score": None,
                "points": clean_pts,
                "group_id": None,
                "description": "",
                "difficult": False,
                "shape_type": "polygon",
                "flags": {},
                "attributes": {},
            }
        )
    return shapes


def build_json_dict(img_filename: str, width: int, height: int, shapes: list[dict]) -> dict:
    return {
        "version": "2.5.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": img_filename,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }


def populate_folder(img_paths: list[Path], target_dir: Path) -> int:
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "classes.txt").write_text("copa\nfiacao\nposte\n", encoding="utf-8")

    count = 0
    for img_p in sorted(img_paths):
        with Image.open(img_p) as im:
            w, h = im.size
        txt_p = PRELABELS_DIR / (img_p.stem + ".txt")
        shapes = yolo_seg_to_xanylabeling_shapes(txt_p, w, h)

        dst_img = target_dir / img_p.name
        dst_json = target_dir / (img_p.stem + ".json")

        shutil.copy2(img_p, dst_img)
        dst_json.write_text(
            json.dumps(build_json_dict(img_p.name, w, h, shapes), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Prepara pasta nativa para o X-AnyLabeling")
    parser.add_argument("--out", type=str, default=str(DEFAULT_WORKSPACE), help="Pasta de destino")
    parser.add_argument(
        "--split-packs",
        type=int,
        default=1,
        help="Se > 1, divide as árvores igualmente em subpastas pacote_1, pacote_2... para colegas do grupo",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    all_imgs = sorted(IMAGES_DIR.glob("*.jpg"))
    if not all_imgs:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em {IMAGES_DIR}")

    if args.split_packs <= 1:
        n = populate_folder(all_imgs, out_dir)
        print(f"Workspace X-AnyLabeling pronto em: {out_dir}")
        print(f"Total preparado: {n} imagens + {n} arquivos .json pré-anotados + classes.txt")
    else:
        # Agrupar por tree_id para não separar ângulos da mesma árvore em pacotes diferentes
        by_tree: dict[str, list[Path]] = {}
        for p in all_imgs:
            parts = p.stem.split("_")
            tid = parts[1] if len(parts) >= 2 else p.stem
            by_tree.setdefault(tid, []).append(p)

        tree_ids = sorted(by_tree.keys())
        chunk = math.ceil(len(tree_ids) / args.split_packs)
        for idx in range(args.split_packs):
            sub_ids = tree_ids[idx * chunk : (idx + 1) * chunk]
            if not sub_ids:
                continue
            sub_imgs = [im for tid in sub_ids for im in by_tree[tid]]
            pack_dir = out_dir / f"pacote_colega_{idx + 1}"
            n = populate_folder(sub_imgs, pack_dir)
            print(f"Pacote {idx + 1}: {n} imagens ({len(sub_ids)} árvores) salvo em {pack_dir}")


if __name__ == "__main__":
    main()

