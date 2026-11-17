"""
Pipeline Operacional Idempotente do ArborGrid Vision (Coleta Territorial + Inferência V3 + Sync WebApp).

Fluxo declarativo em 4 estágios:
  1. Lê definições de regiões (`dataset/sp_geosampa_api/regions.json`)
  2. Constrói / Atualiza o `manifest.json` via WFS GeoSampa (`geoportal:arvore`)
  3. Baixa apenas as imagens Street View faltantes em `dataset/sp_geosampa_api/images_raw/`
  4. Executa a Inferência V3 (SegFormer-B2 1024px via PyTorch GPU `.pt` ou fallback automático para ONNX INT8 `.onnx`)
     com morfologia calibrada (`r1=6px`, `r2=24px`), consenso de paralaxe (`v3_analysis` / `v3_triage`) e sincronização com `webapp/data/manifest.json`.

Exemplos de uso:
  python scripts/run_operational_pipeline.py --status
  python scripts/run_operational_pipeline.py --infer-v3 --sync-webapp
  python scripts/run_operational_pipeline.py --region pinheiros --add-trees 20 --download-missing --infer-v3 --sync-webapp
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import requests


def _load_env_file(env_path: Path) -> None:
    """Carrega variáveis de ambiente de um arquivo .env sem dependências externas."""
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from vision_utils import (
    analyze_view_v3_manifest,
    consolidate_tree_v3_manifest,
    infer_masks,
    load_onnx_session,
    load_segformer_weights,
    render_overlay,
)

DATASET_DIR = BASE_DIR / "dataset" / "sp_geosampa_api"
IMAGES_RAW_DIR = DATASET_DIR / "images"
IMAGES_OVERLAY_DIR = DATASET_DIR / "v3_overlays"
REGIONS_PATH = DATASET_DIR / "regions.json"
MANIFEST_PATH = DATASET_DIR / "manifest.json"
WEBAPP_MANIFEST_PATH = BASE_DIR.parent / "webapp" / "data" / "manifest.json"

PT_CKPT_PATH = BASE_DIR / "models" / "arborgrid_segformer_b2_1024_v3_gold.pt"
ONNX_INT8_PATH = BASE_DIR / "models" / "segformer_v3_int8_dyn_matmul.onnx"

GEOSAMPA_WFS_URL = "http://wfs.geosampa.prefeitura.sp.gov.br/geoserver/geoportal/wfs"
SV_METADATA_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"
SV_IMAGE_URL = "https://maps.googleapis.com/maps/api/streetview"


# ── Helpers Geodésicos e WFS GeoSampa ────────────────────────────────────────
def _get_utm_transformers():
    from pyproj import Transformer

    utm_to_wgs84 = Transformer.from_crs("EPSG:31983", "EPSG:4326", always_xy=True)
    wgs84_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:31983", always_xy=True)
    return utm_to_wgs84, wgs84_to_utm


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    dlon_r = math.radians(lon2 - lon1)
    x = math.sin(dlon_r) * math.cos(lat2_r)
    y = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon_r)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def step_latlon(lat: float, lon: float, bearing_deg: float, step_meters: float) -> tuple[float, float]:
    R = 6378137.0
    brng = math.radians(bearing_deg)
    dlat = (step_meters * math.cos(brng)) / R
    dlon = (step_meters * math.sin(brng)) / (R * math.cos(math.radians(lat)))
    return lat + math.degrees(dlat), lon + math.degrees(dlon)


def fetch_geosampa_trees_in_region(
    center_lat: float, center_lon: float, radius_m: int = 450, max_features: int = 80
) -> list[dict[str, Any]]:
    utm_to_wgs84, wgs84_to_utm = _get_utm_transformers()
    cx, cy = wgs84_to_utm.transform(center_lon, center_lat)
    bbox = f"{cx - radius_m},{cy - radius_m},{cx + radius_m},{cy + radius_m},EPSG:31983"
    params = {
        "service": "WFS",
        "version": "1.0.0",
        "request": "GetFeature",
        "typeName": "geoportal:arvore",
        "outputFormat": "application/json",
        "bbox": bbox,
        "maxFeatures": max_features,
    }
    resp = requests.get(GEOSAMPA_WFS_URL, params=params, timeout=30)
    resp.raise_for_status()
    features = resp.json().get("features", [])
    trees = []
    for feat in features:
        props = feat.get("properties", {})
        geom = feat.get("geometry", {})
        coords = geom.get("coordinates", [])
        if not coords:
            continue
        if geom.get("type") == "MultiPoint":
            ux, uy = coords[0][0], coords[0][1]
        else:
            ux, uy = coords[0], coords[1]
        lon, lat = utm_to_wgs84.transform(ux, uy)
        trees.append(
            {
                "geosampa_id": str(props.get("cd_identif", feat.get("id", "unknown"))),
                "distrito": props.get("nm_distrit") or "SAO PAULO",
                "subprefeitura": props.get("nm_subpref") or "SAO PAULO",
                "tree_lat": round(lat, 8),
                "tree_lon": round(lon, 8),
                "sisgau_last_inspection": props.get("dt_atualiz"),
            }
        )
    return trees


# ── Estágio 1 e 2: Leitura de Regiões e Expansão do Manifest ─────────────────
def load_regions_and_manifest() -> tuple[dict[str, Any], dict[str, Any]]:
    regions_cfg = json.loads(REGIONS_PATH.read_text(encoding="utf-8"))
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    else:
        manifest = {
            "project": "ArborGrid - FIAP Global Solution CitizenTech (4ES)",
            "generated_at_utc": datetime.now().isoformat(),
            "total_trees_collected": 0,
            "total_static_api_images": 0,
            "trees": [],
        }
    return regions_cfg, manifest


def expand_manifest_from_wfs(
    regions_cfg: dict[str, Any],
    manifest: dict[str, Any],
    target_region: str | None,
    add_trees_count: int,
    api_key: str,
) -> int:
    existing_ids = {str(t.get("geosampa_id", "")) for t in manifest.get("trees", [])}
    sv_cfg = regions_cfg.get("streetview_defaults", {})
    step_m = float(sv_cfg.get("parallax_step_m", 10.0))
    added_total = 0

    for rid, reg in regions_cfg.get("regions", {}).items():
        if target_region and rid != target_region:
            continue
        print(f"[WFS] Consultando GeoSampa para região '{rid}' ({reg['label']})...")
        bbox = reg.get("bbox", [-46.696, -23.568, -46.679, -23.555])
        center_lon = (bbox[0] + bbox[2]) / 2.0
        center_lat = (bbox[1] + bbox[3]) / 2.0
        candidates = fetch_geosampa_trees_in_region(
            center_lat,
            center_lon,
            radius_m=int(reg.get("wfs_radius_m", 650)),
            max_features=max(add_trees_count * 3, 60),
        )
        added_reg = 0
        for cand in candidates:
            if added_reg >= add_trees_count:
                break
            tid = cand["geosampa_id"]
            if tid in existing_ids:
                continue

            meta_resp = requests.get(
                SV_METADATA_URL,
                params={
                    "location": f"{cand['tree_lat']},{cand['tree_lon']}",
                    "radius": sv_cfg.get("search_radius_m", 30),
                    "source": "outdoor",
                    "key": api_key,
                },
                timeout=15,
            ).json()
            if meta_resp.get("status") != "OK":
                continue

            pano_lat = meta_resp["location"]["lat"]
            pano_lon = meta_resp["location"]["lng"]
            dist_m = round(haversine_m(pano_lat, pano_lon, cand["tree_lat"], cand["tree_lon"]), 2)
            base_heading = round(calculate_bearing(pano_lat, pano_lon, cand["tree_lat"], cand["tree_lon"]), 2)
            h_int = int(round(base_heading))

            views = [
                {
                    "perspective": "center_front",
                    "pano_id": meta_resp.get("pano_id"),
                    "capture_date": meta_resp.get("date"),
                    "cam_lat": pano_lat,
                    "cam_lon": pano_lon,
                    "distance_m": dist_m,
                    "heading": base_heading,
                    "pitch": 18.0,
                    "fov": 62,
                    "google_maps_url": f"https://www.google.com/maps/@?api=1&map_action=pano&pano={meta_resp.get('pano_id')}&heading={base_heading}&pitch=18.0&fov=62",
                    "file_name": f"tree_{tid}_center_front_h{h_int}.jpg",
                }
            ]

            s_lat, s_lon = step_latlon(pano_lat, pano_lon, (base_heading + 90.0) % 360.0, step_m)
            s_meta = requests.get(
                SV_METADATA_URL,
                params={"location": f"{s_lat},{s_lon}", "radius": 20, "source": "outdoor", "key": api_key},
                timeout=15,
            ).json()
            if s_meta.get("status") == "OK" and s_meta.get("pano_id") != meta_resp.get("pano_id"):
                slat, slon = s_meta["location"]["lat"], s_meta["location"]["lng"]
                s_head = round(calculate_bearing(slat, slon, cand["tree_lat"], cand["tree_lon"]), 2)
                s_dist = round(haversine_m(slat, slon, cand["tree_lat"], cand["tree_lon"]), 2)
                sh_int = int(round(s_head))
                views.append(
                    {
                        "perspective": "side_angle_A",
                        "pano_id": s_meta.get("pano_id"),
                        "capture_date": s_meta.get("date"),
                        "cam_lat": slat,
                        "cam_lon": slon,
                        "distance_m": s_dist,
                        "heading": s_head,
                        "pitch": 15.0,
                        "fov": 55,
                        "google_maps_url": f"https://www.google.com/maps/@?api=1&map_action=pano&pano={s_meta.get('pano_id')}&heading={s_head}&pitch=15.0&fov=55",
                        "file_name": f"tree_{tid}_side_angle_A_h{sh_int}.jpg",
                    }
                )

            cand["region_key"] = rid
            cand["region_label"] = reg["label"]
            cand["dataset_role"] = "blind_control" if reg.get("grid_type") == "underground_control" else "field_inference"
            cand["views"] = views
            manifest["trees"].append(cand)
            existing_ids.add(tid)
            added_reg += 1
            added_total += 1

        print(f"  -> +{added_reg} novas árvores adicionadas em '{rid}'.")

    manifest["total_trees_collected"] = len(manifest["trees"])
    manifest["total_static_api_images"] = sum(len(t.get("views", [])) for t in manifest["trees"])
    return added_total


# ── Estágio 3: Download Incremental de Imagens Street View Faltantes ─────────
def download_missing_views(manifest: dict[str, Any], api_key: str, target_region: str | None = None) -> int:
    IMAGES_RAW_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    for tree in manifest.get("trees", []):
        if target_region and tree.get("region_key") != target_region:
            continue
        for v in tree.get("views", []):
            fname = v.get("file_name")
            if not fname:
                continue
            img_path = IMAGES_RAW_DIR / fname
            if img_path.exists() and img_path.stat().st_size > 1024:
                continue
            params = {
                "size": "640x640",
                "heading": v["heading"],
                "pitch": v.get("pitch", 15.0),
                "fov": v.get("fov", 60),
                "source": "outdoor",
                "key": api_key,
            }
            if v.get("pano_id"):
                params["pano"] = v["pano_id"]
            else:
                params["location"] = f"{v['cam_lat']},{v['cam_lon']}"

            resp = requests.get(SV_IMAGE_URL, params=params, timeout=20)
            if resp.status_code == 200 and len(resp.content) > 1024:
                img_path.write_bytes(resp.content)
                downloaded += 1
                time.sleep(0.05)
    return downloaded


# ── Estágio 4: Inferência Territorial V3 (PyTorch GPU ou ONNX INT8 CPU) ──────
def load_best_available_engine() -> tuple[Any, Any, str]:
    """Carrega `.pt` na GPU (se disponível) ou faz fallback transparente para `.onnx` INT8."""
    try:
        import torch

        if PT_CKPT_PATH.exists() and torch.cuda.is_available():
            device = torch.device("cuda")
            model = load_segformer_weights(PT_CKPT_PATH, device)
            return model, device, f"PyTorch GPU ({PT_CKPT_PATH.name})"
    except Exception:
        pass

    if ONNX_INT8_PATH.exists():
        sess = load_onnx_session(ONNX_INT8_PATH)
        return sess, None, f"ONNX Runtime INT8 CPU ({ONNX_INT8_PATH.name})"

    if PT_CKPT_PATH.exists():
        import torch

        device = torch.device("cpu")
        model = load_segformer_weights(PT_CKPT_PATH, device)
        return model, device, f"PyTorch CPU ({PT_CKPT_PATH.name})"

    raise FileNotFoundError(
        f"Nenhum peso V3 encontrado em {PT_CKPT_PATH} ou {ONNX_INT8_PATH}."
    )


def run_v3_inference_on_manifest(
    manifest: dict[str, Any],
    recompute_all: bool = False,
    target_region: str | None = None,
) -> dict[str, Any]:
    engine, device, engine_label = load_best_available_engine()
    print(f"[INFERÊNCIA V3] Motor ativo: {engine_label}")
    IMAGES_OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

    inferred_views = 0
    t0 = time.perf_counter()

    for tree in manifest.get("trees", []):
        if target_region and tree.get("region_key") != target_region:
            continue

        views_updated = False
        for v in tree.get("views", []):
            if not recompute_all and "v3_analysis" in v:
                continue
            fname = v.get("file_name")
            if not fname:
                continue
            img_path = IMAGES_RAW_DIR / fname
            if not img_path.exists():
                continue
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue

            mask_tree, mask_wire = infer_masks(engine, img_bgr, device=device, res=1024)
            v3_dict, _, _ = analyze_view_v3_manifest(mask_tree, mask_wire, fname)
            v["v3_analysis"] = v3_dict

            # Salva overlay V3
            ov_img = render_overlay(img_bgr, mask_tree, mask_wire, alpha_tree=0.36, alpha_wire=0.25)
            cv2.imwrite(str(IMAGES_OVERLAY_DIR / v3_dict["v3_overlay_file"]), ov_img, [cv2.IMWRITE_JPEG_QUALITY, 90])

            inferred_views += 1
            views_updated = True

        if views_updated or recompute_all or "v3_triage" not in tree:
            tree["v3_triage"] = consolidate_tree_v3_manifest(tree.get("views", []))

    status_counts = Counter(
        t.get("v3_triage", {}).get("consolidated", "PENDENTE")
        for t in manifest.get("trees", [])
    )
    elapsed = time.perf_counter() - t0
    print(
        f"[INFERÊNCIA V3] Concluída: {inferred_views} visadas processadas em {elapsed:.1f}s | "
        f"Distribuição: {dict(status_counts)}"
    )
    return dict(status_counts)


# ── Relatório de Status Operacional ──────────────────────────────────────────
def print_pipeline_status(regions_cfg: dict[str, Any], manifest: dict[str, Any]) -> None:
    trees = manifest.get("trees", [])
    total_views = sum(len(t.get("views", [])) for t in trees)
    existing_files = sum(
        1
        for t in trees
        for v in t.get("views", [])
        if (IMAGES_RAW_DIR / v.get("file_name", "")).exists()
    )
    inferred_views = sum(
        1 for t in trees for v in t.get("views", []) if "v3_analysis" in v
    )
    parallax_div_count = sum(
        1 for t in trees if t.get("v3_triage", {}).get("has_paralax_divergence")
    )

    print("=" * 90)
    print("ARBORGRID VISION — STATUS DA PIPELINE OPERACIONAL (GEOSAMPA WFS + STREET VIEW + V3)")
    print("=" * 90)
    print(f"Arquivo de Regiões : {REGIONS_PATH.relative_to(BASE_DIR)} ({len(regions_cfg.get('regions', []))} zonas configuradas)")
    print(f"Manifest Principal : {MANIFEST_PATH.relative_to(BASE_DIR)}")
    print(f"Árvores Mapeadas   : {len(trees)} árvores | {existing_files}/{total_views} imagens em disco | {inferred_views}/{total_views} com inferência V3")
    print(f"Pesos Disponíveis  : PyTorch (.pt)={'SIM' if PT_CKPT_PATH.exists() else 'NÃO'} | ONNX INT8 (.onnx)={'SIM' if ONNX_INT8_PATH.exists() else 'NÃO'}")
    print(f"Paralaxe Multi-View: {parallax_div_count} árvores com divergência angular auditada pelo consenso")
    print("-" * 92)
    print(f"{'Região':<20} | {'Árvores':>7} | {'Imagens':>7} | {'Crítico':>8} | {'Paralaxe':>8} | {'Proximid.':>9} | {'Seguro':>7}")
    print("-" * 92)

    for rid, reg in regions_cfg.get("regions", {}).items():
        rtrees = [t for t in trees if t.get("region_key") == rid]
        rviews = sum(len(t.get("views", [])) for t in rtrees)
        sc = Counter(t.get("v3_triage", {}).get("consolidated", "PENDENTE") for t in rtrees)
        crit = sc["CRITICO_CONFIRMADO"] + sc["CRITICO_MONO_VISADA"]
        para = sc["DIVERGENCIA_PARALAXE"]
        prox = sc["PROXIMIDADE"] + sc["PROXIMIDADE_MONO_VISADA"]
        seg = sc["SEGURO"] + sc["SEGURO_MONO_VISADA"]
        print(
            f"{rid:<20} | {len(rtrees):>7} | {rviews:>7} | "
            f"{crit:>8} | {para:>8} | {prox:>9} | {seg:>7}"
        )

    print("-" * 92)
    tot_sc = Counter(t.get("v3_triage", {}).get("consolidated", "PENDENTE") for t in trees)
    t_crit = tot_sc["CRITICO_CONFIRMADO"] + tot_sc["CRITICO_MONO_VISADA"]
    t_para = tot_sc["DIVERGENCIA_PARALAXE"]
    t_prox = tot_sc["PROXIMIDADE"] + tot_sc["PROXIMIDADE_MONO_VISADA"]
    t_seg = tot_sc["SEGURO"] + tot_sc["SEGURO_MONO_VISADA"]
    print(
        f"{'TOTAL GERAL':<20} | {len(trees):>7} | {total_views:>7} | "
        f"{t_crit:>8} | {t_para:>8} | {t_prox:>9} | {t_seg:>7}"
    )
    print("=" * 92)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline Operacional do ArborGrid Vision (GeoSampa WFS -> Street View -> Inferência V3 -> WebApp)."
    )
    parser.add_argument("--status", action="store_true", help="Exibe o diagnóstico completo do manifest e das 8 regiões.")
    parser.add_argument("--region", type=str, default=None, help="Filtra execução para uma região específica de regions.json.")
    parser.add_argument("--add-trees", type=int, default=0, help="Consulta o GeoSampa WFS e adiciona até N novas árvores por região.")
    parser.add_argument("--download-missing", action="store_true", help="Baixa do Google Street View apenas as imagens ausentes no disco.")
    parser.add_argument("--infer-v3", action="store_true", help="Executa a inferência V3 + morfologia nas visadas pendentes.")
    parser.add_argument("--recompute-inference", action="store_true", help="Força o recálculo da inferência V3 em todas as visadas.")
    parser.add_argument("--sync-webapp", action="store_true", help="Sincroniza o manifest atualizado com ../webapp/data/manifest.json.")
    args = parser.parse_args()

    _load_env_file(BASE_DIR / ".env")
    api_key = os.getenv("GOOGLE_MAPS_API_KEY", "")

    regions_cfg, manifest = load_regions_and_manifest()
    modified = False

    if args.add_trees > 0:
        if not api_key:
            raise RuntimeError("GOOGLE_MAPS_API_KEY não configurada no .env para consulta de Street View Metadata.")
        added = expand_manifest_from_wfs(regions_cfg, manifest, args.region, args.add_trees, api_key)
        if added > 0:
            modified = True

    if args.download_missing:
        if not api_key:
            raise RuntimeError("GOOGLE_MAPS_API_KEY não configurada no .env para download de imagens Street View.")
        dl = download_missing_views(manifest, api_key, target_region=args.region)
        print(f"[STREET VIEW] {dl} novas imagens baixadas.")

    if args.infer_v3 or args.recompute_inference:
        run_v3_inference_on_manifest(
            manifest, recompute_all=args.recompute_inference, target_region=args.region
        )
        modified = True

    if modified:
        MANIFEST_PATH.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[MANIFEST] Salvo em {MANIFEST_PATH.relative_to(BASE_DIR)}.")

    if args.sync_webapp:
        if WEBAPP_MANIFEST_PATH.parent.exists():
            shutil.copy2(MANIFEST_PATH, WEBAPP_MANIFEST_PATH)
            print(f"[SYNC WEBAPP] Sincronizado com {WEBAPP_MANIFEST_PATH}.")

    if args.status or not (
        args.add_trees or args.download_missing or args.infer_v3 or args.recompute_inference or args.sync_webapp
    ):
        print_pipeline_status(regions_cfg, manifest)


if __name__ == "__main__":
    main()
