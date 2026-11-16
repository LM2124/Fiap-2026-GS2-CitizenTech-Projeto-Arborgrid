"""Gera mosaicos visuais de auditoria a partir do manifest.json (amostra geral + top conflitos GPU)."""
import json
import math
from pathlib import Path
from PIL import Image, ImageDraw

BASE = Path(__file__).resolve().parent.parent / "dataset" / "sp_geosampa_api"
manifest = json.loads((BASE / "manifest.json").read_text(encoding="utf-8"))

# Mosaico dos candidatos de conflito detectados na GPU
conf_views = []
for t in manifest["trees"]:
    if t.get("consolidated_triage") == "CANDIDATO_CONFLITO":
        best_v = max(t["views"], key=lambda v: (v.get("gpu_analysis") or {}).get("conflict_overlap_px", 0))
        conf_views.append((t["geosampa_id"], best_v))

if conf_views:
    conf_views = conf_views[:8]
    cols = min(4, len(conf_views))
    rows = math.ceil(len(conf_views) / cols)
    mosaic = Image.new("RGB", (cols * 480, rows * 480), (20, 20, 20))
    for i, (tid, v) in enumerate(conf_views):
        ga = v.get("gpu_analysis") or {}
        ov = BASE / "gpu_overlays" / ga["overlay_file"]
        im = Image.open(ov).convert("RGB").resize((480, 480))
        draw = ImageDraw.Draw(im)
        draw.rectangle([0, 0, 480, 32], fill=(175, 25, 25))
        txt = f"ID #{tid} | intersecao={ga.get('conflict_overlap_px', 0)}px | {v.get('capture_date')}"
        draw.text((8, 9), txt, fill=(255, 255, 255))
        r, c = divmod(i, cols)
        mosaic.paste(im, (c * 480, r * 480))
    out_c = BASE / "mosaico_top_conflitos_pinheiros.jpg"
    mosaic.save(out_c, quality=90)
    print(f"Mosaico de conflitos salvo em: {out_c}")
