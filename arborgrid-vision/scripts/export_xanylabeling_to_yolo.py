"""
ArborGrid - Consolidador de Pacotes & Exportador Seguro de X-AnyLabeling (.json) para YOLO-Seg
Módulo: arborgrid-vision / scripts / export_xanylabeling_to_yolo.py

Recursos "à prova de erro":
  - Lê apenas os arquivos consolidados na raiz de `--src` (`src_dir.glob("*.json")`), evitando duplicar
    backups ou partições em subpastas (`pacote_1_prelabel`, `pacote_1`, `pacote_2`, etc.).
  - Suporta `--merge-from <pasta>` para consolidar pacotes curados na raiz do workspace
    (respeitando descartes em `_delete_/`).
  - Auto-espessamento de segurança para `fiacao`: se um humano desenhou um polígono de fio fino demais
    (< 3.5 px de espessura), expande morfologicamente para ~5 px para não sumir no downsampling 1/4 (160x160) do YOLOv8-Seg.
  - Divisão Treino (80%) / Validação (20%) agrupada por ID da árvore (zero Data Leakage entre ângulos).
"""

import json
import random
import shutil
import argparse
import cv2
import numpy as np
from pathlib import Path

CV_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SRC = CV_ROOT / "dataset" / "workspace_xanylabeling"
DEFAULT_OUT = CV_ROOT / "dataset" / "yolo_seg_ready"

LABEL_NORMALIZATION = {
    "copa": 0,
    "copas": 0,
    "arvore": 0,
    "árvore": 0,
    "fiacao": 1,
    "fiação": 1,
    "fio": 1,
    "fios": 1,
    "cabo": 1,
    "cabos": 1,
    "poste": 2,
    "postes": 2,
}


def merge_pack_into_workspace(pack_dir: Path, workspace_root: Path) -> dict:
    """
    Copia os .json revisados de `pack_dir` para `workspace_root` (sobrescrevendo o pré-label original).
    Se alguma imagem foi movida para `pack_dir/_delete_`, move também o par na raiz para `workspace_root/_delete_`.
    """
    if not pack_dir.exists():
        raise FileNotFoundError(f"Pacote não encontrado: {pack_dir}")

    updated = 0
    deleted = 0

    # 1. Propaga descartes (_delete_) se existirem no pacote revisado
    del_src = pack_dir / "_delete_"
    if del_src.exists():
        del_dst = workspace_root / "_delete_"
        del_dst.mkdir(parents=True, exist_ok=True)
        for f in del_src.iterdir():
            if f.is_file():
                stem = f.stem
                for ext in (".jpg", ".json", ".png"):
                    root_file = workspace_root / f"{stem}{ext}"
                    if root_file.exists():
                        shutil.move(str(root_file), str(del_dst / root_file.name))
                        if ext == ".json":
                            deleted += 1

    # 2. Sobrescreve os .json na raiz do workspace com os .json revisados do pacote
    for jf in sorted(pack_dir.glob("*.json")):
        if jf.name == "manifest.json":
            continue
        shutil.copy2(jf, workspace_root / jf.name)
        updated += 1

    return {"updated_jsons": updated, "propagated_deletes": deleted}


def ensure_min_wire_thickness(pts: list[list[float]], w: int, h: int, min_thick_px: float = 3.5) -> list[list[float]]:
    """Garante que polígonos de fiação muito finos tenham espessura >= ~5px para sobreviver à máscara 160x160 do YOLO-Seg."""
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


def convert_json_to_yolo_lines(json_path: Path) -> tuple[list[str], list[str], bool, int]:
    """Retorna (linhas_yolo, avisos, descartada_por_flag, qtd_fios_espessados)."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    flags = data.get("flags") or {}
    if flags.get("descartar") is True:
        return [], [], True, 0

    w = int(data.get("imageWidth") or 640)
    h = int(data.get("imageHeight") or 640)
    lines = []
    warnings = []
    thickened = 0

    for sh in data.get("shapes") or []:
        raw_lbl = str(sh.get("label") or "").strip().lower()
        if raw_lbl not in LABEL_NORMALIZATION:
            warnings.append(f"{json_path.name}: etiqueta desconhecida '{sh.get('label')}' ignorada!")
            continue

        cid = LABEL_NORMALIZATION[raw_lbl]
        pts = sh.get("points") or []
        if len(pts) < 3:
            continue

        if cid == 1:
            new_pts = ensure_min_wire_thickness(pts, w, h)
            if new_pts is not pts:
                thickened += 1
                pts = new_pts

        coords = []
        for x, y in pts:
            nx = min(1.0, max(0.0, float(x) / float(w)))
            ny = min(1.0, max(0.0, float(y) / float(h)))
            coords.append(f"{nx:.6f} {ny:.6f}")
        lines.append(f"{cid} " + " ".join(coords))

    return lines, warnings, False, thickened


def main():
    parser = argparse.ArgumentParser(description="Consolida pacotes e converte X-AnyLabeling para YOLO-Seg final")
    parser.add_argument("--src", type=str, default=str(DEFAULT_SRC), help="Pasta raiz do workspace com .jpg e .json")
    parser.add_argument(
        "--merge-from",
        type=str,
        nargs="*",
        default=None,
        help="Lista opcional de subpastas revisadas (ex: pacote_1 pacote_2 pacote_3_hard_negatives) para consolidar na raiz antes de exportar",
    )
    parser.add_argument("--merge-only", action="store_true", help="Apenas consolida os pacotes na raiz sem exportar para YOLO ainda")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT), help="Pasta final do dataset YOLO-Seg")
    parser.add_argument("--val-ratio", type=float, default=0.20, help="Proporção de validação por árvore")
    parser.add_argument("--seed", type=int, default=42, help="Semente reprodutível")
    args = parser.parse_args()

    src_dir = Path(args.src)
    out_dir = Path(args.out)

    if args.merge_from:
        for pack_arg in args.merge_from:
            p_path = Path(pack_arg)
            if not p_path.is_absolute():
                cand = src_dir / pack_arg
                p_path = cand if cand.exists() else Path(pack_arg)
            res = merge_pack_into_workspace(p_path, src_dir)
            print(f"[MERGE OK] {p_path.name} -> {src_dir.name}: {res['updated_jsons']} JSONs sobrescritos, {res['propagated_deletes']} descartes aplicados.")

    if args.merge_only:
        return

    # Lê APENAS os .json do nível raiz de src_dir (nunca mistura subpastas de backup/pacotes!)
    json_files = [
        p for p in sorted(src_dir.glob("*.json"))
        if p.name != "manifest.json"
    ]
    if not json_files:
        raise FileNotFoundError(f"Nenhum arquivo .json encontrado no nível raiz de {src_dir}")

    # Agrupar por tree_id para split seguro sem vazamento entre perspectivas
    by_tree: dict[str, list[Path]] = {}
    for jp in json_files:
        parts = jp.stem.split("_")
        tid = parts[1] if len(parts) >= 2 and parts[0] == "tree" else jp.stem
        by_tree.setdefault(tid, []).append(jp)

    tree_ids = sorted(by_tree.keys())
    rng = random.Random(args.seed)
    rng.shuffle(tree_ids)

    n_val = max(1, round(len(tree_ids) * args.val_ratio))
    val_trees = set(tree_ids[:n_val])

    if out_dir.exists():
        shutil.rmtree(out_dir)

    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0, "skipped": 0, "thickened_wires": 0}
    all_warnings = []

    for tid, jpaths in by_tree.items():
        split = "val" if tid in val_trees else "train"
        for jp in jpaths:
            img_p = jp.with_suffix(".jpg")
            if not img_p.exists():
                for ext in (".png", ".jpeg"):
                    cand = jp.with_suffix(ext)
                    if cand.exists():
                        img_p = cand
                        break
            if not img_p.exists():
                counts["skipped"] += 1
                continue

            lines, warns, discarded, thick = convert_json_to_yolo_lines(jp)
            all_warnings.extend(warns)
            counts["thickened_wires"] += thick
            if discarded:
                counts["skipped"] += 1
                continue

            shutil.copy2(img_p, out_dir / "images" / split / img_p.name)
            (out_dir / "labels" / split / (jp.stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")
            counts[split] += 1

    yaml_text = (
        f"path: {out_dir.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n\n"
        f"nc: 3\n"
        f"names:\n"
        f"  0: copa\n"
        f"  1: fiacao\n"
        f"  2: poste\n"
    )
    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(yaml_text, encoding="utf-8")

    print("=" * 68)
    print("DATASET YOLO-SEG EXPORTADO COM SUCESSO (SEM DATA LEAKAGE)")
    print(f"  -> Imagens em Treino          : {counts['train']}")
    print(f"  -> Imagens em Validação       : {counts['val']}")
    print(f"  -> Descartadas/Puladas        : {counts['skipped']}")
    print(f"  -> Fios finos auto-espessados : {counts['thickened_wires']}")
    print(f"  -> Configuração salva         : {yaml_path}")
    print("=" * 68)
    if all_warnings:
        print("\n[AVISOS DE ETIQUETAS]:")
        for w in all_warnings:
            print("  !", w)


if __name__ == "__main__":
    main()
