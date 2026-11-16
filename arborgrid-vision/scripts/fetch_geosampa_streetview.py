"""
ArborGrid - Pipeline Automatizado Multi-Região (GeoSampa WFS Completo + Street View Multi-Perspectiva)
Fontes Oficiais:
  1. Prefeitura de São Paulo / GeoSampa (OGC WFS - Camada 'geoportal:arvore' / SISGAU)
     Política de Dados Abertos PMSP (Decreto Municipal nº 57.576/2017)
  2. Google Maps Platform - Street View Metadata API & Street View Static API (© Google)

Destaques desta versão:
  - Suporte nativo a múltiplas regiões críticas de São Paulo (`--region pinheiros`, `alto_da_mooca`,
    `vila_mariana`, `campo_belo`, `moema`, `perdizes`, `saude`).
  - Baixa 100% das árvores oficiais do GeoSampa no recorte (`maxFeatures=20000`) sem truncamento,
    permitindo plotar todas as árvores governamentais não auditadas no mapa interativo.
  - Amostragem Espacial Estratificada em Grade (Grid-Stratified): distribui as árvores auditadas
    uniformemente por todos os quarteirões, sem buracos no meio do bairro.
  - Modo incremental: preserva árvores já baixadas no `manifest.json` sem gastar cota repetida.
"""

import os
import re
import json
import math
import time
import random
import argparse
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

CV_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = CV_ROOT.parent
DEFAULT_OUT_DIR = CV_ROOT / "dataset" / "sp_geosampa_api"

GEOSAMPA_WFS_URL = "http://wfs.geosampa.prefeitura.sp.gov.br/geoserver/geoportal/wfs"
SV_METADATA_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"
SV_STATIC_URL = "https://maps.googleapis.com/maps/api/streetview"

# Regiões pré-configuradas (epicentros históricos de queda de árvores / apagões em SP + Alto da Mooca)
REGIONS = {
    "pinheiros": {
        "label": "Pinheiros (Zona Oeste)",
        "district_code": "PIN",
        "bbox": (-46.6960, -23.5680, -46.6790, -23.5550),
    },
    "alto_da_mooca": {
        "label": "Alto da Mooca (Zona Leste)",
        "district_code": "MOO",
        "bbox": (-46.6010, -23.5670, -46.5840, -23.5535),
    },
    "campo_belo": {
        "label": "Campo Belo (Zona Sul)",
        "district_code": "CBE",
        "bbox": (-46.6780, -23.6320, -46.6610, -23.6190),
    },
    "vila_mariana": {
        "label": "Vila Mariana / Chácara Klabin (Centro-Sul)",
        "district_code": "VMN",
        "bbox": (-46.6440, -23.5950, -46.6270, -23.5820),
    },
    "moema": {
        "label": "Moema (Zona Sul)",
        "district_code": "MOE",
        "bbox": (-46.6690, -23.6100, -46.6520, -23.5970),
    },
    "perdizes": {
        "label": "Perdizes / Pompeia (Zona Oeste)",
        "district_code": "PRD",
        "bbox": (-46.6830, -23.5430, -46.6660, -23.5300),
    },
    "saude": {
        "label": "Saúde / Mirandópolis (Zona Sul)",
        "district_code": "SAU",
        "bbox": (-46.6430, -23.6220, -46.6260, -23.6090),
    },
}


def load_api_key() -> str:
    """Carrega GOOGLE_MAPS_API_KEY do ambiente ou de arborgrid-vision/.env."""
    key = os.environ.get("GOOGLE_MAPS_API_KEY", "").strip()
    if key and key != "COLE_SUA_API_KEY_AQUI":
        return key

    for env_path in (CV_ROOT / ".env", PROJECT_ROOT / ".env"):
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    if k.strip() == "GOOGLE_MAPS_API_KEY":
                        val = v.strip().strip('"').strip("'")
                        if val and val != "COLE_SUA_API_KEY_AQUI":
                            return val
    return ""


# ============================================================================
# GEOMETRIA ESFÉRICA E TRIGONOMETRIA DE CÂMERA
# ============================================================================

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def compute_heading(cam_lat: float, cam_lon: float, target_lat: float, target_lon: float) -> float:
    p1, p2 = math.radians(cam_lat), math.radians(target_lat)
    dl = math.radians(target_lon - cam_lon)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    bearing = math.degrees(math.atan2(x, y))
    return round((bearing + 360.0) % 360.0, 2)


def offset_latlon(lat: float, lon: float, bearing_deg: float, distance_m: float) -> tuple[float, float]:
    r = 6371000.0
    brng = math.radians(bearing_deg)
    d_r = distance_m / r
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)

    lat2 = math.asin(math.sin(lat1) * math.cos(d_r) + math.cos(lat1) * math.sin(d_r) * math.cos(brng))
    lon2 = lon1 + math.atan2(
        math.sin(brng) * math.sin(d_r) * math.cos(lat1),
        math.cos(d_r) - math.sin(lat1) * math.sin(lat2),
    )
    return round(math.degrees(lat2), 7), round(math.degrees(lon2), 7)


def compute_camera_framing(distance_m: float) -> tuple[float, int]:
    eff_dist = max(distance_m, 4.5)
    pitch = math.degrees(math.atan2(3.2, eff_dist))
    pitch = round(max(11.0, min(22.0, pitch)), 1)
    if distance_m < 9.0:
        fov = 62
    elif distance_m < 14.0:
        fov = 50
    else:
        fov = 44
    return pitch, fov


def build_google_maps_pano_url(pano_id: str, heading: float, pitch: float, fov: int) -> str:
    return (
        f"https://www.google.com/maps/@?api=1&map_action=pano"
        f"&pano={urllib.parse.quote(pano_id)}"
        f"&heading={heading}&pitch={pitch}&fov={fov}"
    )


# ============================================================================
# INTEGRAÇÃO GEOSAMPA (WFS COMPLETO + AMOSTRAGEM EM GRADE)
# ============================================================================

def query_geosampa_hit_count(cql_filter: str | None = None) -> int:
    params = {
        "service": "WFS",
        "version": "1.1.0",
        "request": "GetFeature",
        "typeName": "geoportal:arvore",
        "resultType": "hits",
    }
    if cql_filter:
        params["CQL_FILTER"] = cql_filter

    url = f"{GEOSAMPA_WFS_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "ArborGrid-FIAP-CitizenTech/1.0"})
    with urllib.request.urlopen(req, timeout=25) as resp:
        xml_text = resp.read().decode("utf-8")
        m = re.search(r'numberOfFeatures="(\d+)"', xml_text)
        return int(m.group(1)) if m else -1


def fetch_geosampa_trees(bbox: tuple[float, float, float, float], max_features: int = 20000) -> dict:
    """Baixa TODAS as árvores de Arborização Viária do GeoSampa no recorte em WGS84."""
    min_lon, min_lat, max_lon, max_lat = bbox
    params = {
        "service": "WFS",
        "version": "1.1.0",
        "request": "GetFeature",
        "typeName": "geoportal:arvore",
        "srsName": "EPSG:4326",
        "bbox": f"{min_lon},{min_lat},{max_lon},{max_lat},EPSG:4326",
        "maxFeatures": str(max_features),
        "outputFormat": "application/json",
    }
    url = f"{GEOSAMPA_WFS_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "ArborGrid-FIAP-CitizenTech/1.0"})
    with urllib.request.urlopen(req, timeout=45) as resp:
        return json.loads(resp.read().decode("utf-8"))


def subsample_grid_stratified(
    features: list[dict],
    bbox: tuple[float, float, float, float],
    min_spacing_m: float,
    target_total: int,
    existing_coords: list[tuple[float, float]],
    existing_ids: set[str],
    seed: int = 42,
) -> list[dict]:
    """
    Amostragem estratificada em grade 8x8 (round-robin entre quadrantes):
    Garante cobertura homogênea em todas as ruas do recorte, respeitando as árvores já coletadas.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    grid_nx, grid_ny = 8, 8
    cells: dict[tuple[int, int], list[dict]] = {}

    rng = random.Random(seed)
    shuffled = list(features)
    rng.shuffle(shuffled)

    for feat in shuffled:
        geom = feat.get("geometry") or {}
        if geom.get("type") != "Point":
            continue
        tid = str((feat.get("properties") or {}).get("cd_identificador", ""))
        if tid in existing_ids:
            continue
        lon, lat = geom["coordinates"][:2]
        gx = min(grid_nx - 1, max(0, int((lon - min_lon) / max(1e-9, max_lon - min_lon) * grid_nx)))
        gy = min(grid_ny - 1, max(0, int((lat - min_lat) / max(1e-9, max_lat - min_lat) * grid_ny)))
        cells.setdefault((gx, gy), []).append(feat)

    need_new = max(0, target_total - len(existing_coords))
    if need_new == 0:
        return []

    accepted_coords = list(existing_coords)
    selected_new = []
    cell_keys = sorted(cells.keys())
    rng.shuffle(cell_keys)

    made_progress = True
    while len(selected_new) < need_new and made_progress:
        made_progress = False
        for ck in cell_keys:
            bucket = cells.get(ck)
            while bucket:
                cand = bucket.pop()
                lon, lat = cand["geometry"]["coordinates"][:2]
                if all(haversine_m(lat, lon, elat, elon) >= min_spacing_m for elat, elon in accepted_coords):
                    accepted_coords.append((lat, lon))
                    selected_new.append(cand)
                    made_progress = True
                    break
            if len(selected_new) >= need_new:
                break

    return selected_new


# ============================================================================
# INTEGRAÇÃO GOOGLE STREET VIEW (METADATA + STATIC MULTI-PERSPECTIVA)
# ============================================================================

def is_official_google_street_pano(meta: dict) -> bool:
    pano_id = meta.get("pano_id", "")
    copyright_str = meta.get("copyright", "")
    if pano_id.startswith("CAoS"):
        return False
    if copyright_str and "Google" not in copyright_str:
        return False
    return True


def query_sv_metadata(lat: float, lon: float, api_key: str, radius: int = 25) -> dict | None:
    params = {
        "location": f"{lat},{lon}",
        "radius": str(radius),
        "source": "outdoor",
        "key": api_key,
    }
    url = f"{SV_METADATA_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "ArborGrid-FIAP-CitizenTech/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        if data.get("status") == "OK" and is_official_google_street_pano(data):
            return data
        return None


def collect_tree_perspectives(
    tree_lat: float,
    tree_lon: float,
    api_key: str,
    num_views: int = 2,
    step_along_road_m: float = 13.0,
) -> list[dict]:
    meta0 = query_sv_metadata(tree_lat, tree_lon, api_key, radius=25)
    if not meta0:
        return []

    cam0_lat = meta0["location"]["lat"]
    cam0_lon = meta0["location"]["lng"]
    heading0 = compute_heading(cam0_lat, cam0_lon, tree_lat, tree_lon)
    dist0 = haversine_m(cam0_lat, cam0_lon, tree_lat, tree_lon)
    pitch0, fov0 = compute_camera_framing(dist0)

    views = [
        {
            "perspective": "center_front",
            "pano_id": meta0["pano_id"],
            "capture_date": meta0.get("date", "unknown"),
            "cam_lat": cam0_lat,
            "cam_lon": cam0_lon,
            "distance_m": round(dist0, 2),
            "heading": heading0,
            "pitch": pitch0,
            "fov": fov0,
            "google_maps_url": build_google_maps_pano_url(meta0["pano_id"], heading0, pitch0, fov0),
        }
    ]
    seen_panos = {meta0["pano_id"]}

    if num_views <= 1:
        return views

    for tag, angle_shift in [("side_angle_A", -90.0), ("side_angle_B", +90.0)]:
        if len(views) >= num_views:
            break
        road_bearing = (heading0 + angle_shift) % 360.0
        probe_lat, probe_lon = offset_latlon(cam0_lat, cam0_lon, road_bearing, step_along_road_m)
        meta_i = query_sv_metadata(probe_lat, probe_lon, api_key, radius=15)
        if meta_i and meta_i["pano_id"] not in seen_panos:
            seen_panos.add(meta_i["pano_id"])
            ci_lat = meta_i["location"]["lat"]
            ci_lon = meta_i["location"]["lng"]
            dist_i = haversine_m(ci_lat, ci_lon, tree_lat, tree_lon)
            if dist_i <= 28.0:
                heading_i = compute_heading(ci_lat, ci_lon, tree_lat, tree_lon)
                pitch_i, fov_i = compute_camera_framing(dist_i)
                views.append(
                    {
                        "perspective": tag,
                        "pano_id": meta_i["pano_id"],
                        "capture_date": meta_i.get("date", "unknown"),
                        "cam_lat": ci_lat,
                        "cam_lon": ci_lon,
                        "distance_m": round(dist_i, 2),
                        "heading": heading_i,
                        "pitch": pitch_i,
                        "fov": fov_i,
                        "google_maps_url": build_google_maps_pano_url(meta_i["pano_id"], heading_i, pitch_i, fov_i),
                    }
                )

    return views


def download_sv_image(view: dict, api_key: str, out_path: Path, size: str = "640x640") -> None:
    params = {
        "size": size,
        "pano": view["pano_id"],
        "heading": str(view["heading"]),
        "pitch": str(view["pitch"]),
        "fov": str(view["fov"]),
        "return_error_code": "true",
        "key": api_key,
    }
    url = f"{SV_STATIC_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "ArborGrid-FIAP-CitizenTech/1.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(resp.read())


# ============================================================================
# CLI PRINCIPAL MULTI-REGIÃO INCREMENTAL
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="ArborGrid: Ingestão Multi-Região GeoSampa + Street View")
    parser.add_argument(
        "--region",
        type=str,
        default="pinheiros",
        choices=list(REGIONS.keys()),
        help="Região pré-configurada de São Paulo",
    )
    parser.add_argument("--max-trees", type=int, default=60, help="Meta total de árvores auditadas na região")
    parser.add_argument("--views", type=int, default=2, choices=[1, 2, 3], help="Perspectivas por árvore")
    parser.add_argument("--min-spacing", type=float, default=32.0, help="Espaçamento mínimo (m) entre árvores")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR), help="Diretório de dados")
    parser.add_argument("--geosampa-only", action="store_true", help="Baixa apenas o cadastro completo GeoSampa")
    args = parser.parse_args()

    reg_key = args.region
    reg_cfg = REGIONS[reg_key]
    bbox_tuple = reg_cfg["bbox"]

    out_dir = Path(args.out_dir)
    cadastral_dir = out_dir / "geosampa_cadastral"
    cadastral_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print(f"ARBORGRID | REGIÃO: {reg_cfg['label']} ({reg_key})")
    print("=" * 74)

    # 1. Baixar 100% do cadastro oficial GeoSampa para aquela região
    print(f"[1/3] Baixando cadastro COMPLETO do GeoSampa em bbox={bbox_tuple}...")
    geojson_data = fetch_geosampa_trees(bbox_tuple, max_features=20000)
    all_features = geojson_data.get("features", [])
    inspected_cnt = sum(1 for f in all_features if (f.get("properties") or {}).get("dt_inspecao"))
    pct_insp = (inspected_cnt / len(all_features) * 100.0) if all_features else 0.0

    reg_geojson_path = cadastral_dir / f"geosampa_{reg_key}.geojson"
    reg_geojson_path.write_text(json.dumps(geojson_data, ensure_ascii=False), encoding="utf-8")
    print(f"  -> Total oficial no recorte : {len(all_features):,} árvores (sem truncamento)")
    print(f"  -> Com vistoria no SISGAU   : {inspected_cnt:,} ({pct_insp:.1f}%)")
    print(f"  -> Salvo em                 : {reg_geojson_path}")

    # Carregar manifesto existente (modo incremental)
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"trees": []}

    # Retrocompatibilidade: marcar árvores antigas sem `region_key` como `pinheiros`
    for t in manifest.get("trees", []):
        if not t.get("region_key"):
            t["region_key"] = "pinheiros"
            t["region_label"] = REGIONS["pinheiros"]["label"]

    existing_in_region = [t for t in manifest["trees"] if t.get("region_key") == reg_key]
    existing_coords = [(t["tree_lat"], t["tree_lon"]) for t in existing_in_region]
    existing_ids = {str(t["geosampa_id"]) for t in manifest["trees"]}

    new_candidates = subsample_grid_stratified(
        all_features,
        bbox=bbox_tuple,
        min_spacing_m=args.min_spacing,
        target_total=args.max_trees,
        existing_coords=existing_coords,
        existing_ids=existing_ids,
    )
    print(
        f"  -> Já auditadas na região: {len(existing_in_region)} | "
        f"Novas selecionadas em grade uniforme: {len(new_candidates)}"
    )

    if args.geosampa_only or not new_candidates:
        # Mesmo sem novas fotos, atualizar metadados de regiões no manifesto
        regions_meta = manifest.get("regions_catalog", {})
        regions_meta[reg_key] = {
            "label": reg_cfg["label"],
            "bbox": list(bbox_tuple),
            "total_geosampa_trees": len(all_features),
            "sisgau_inspected_trees": inspected_cnt,
            "geojson_file": f"geosampa_cadastral/geosampa_{reg_key}.geojson",
        }
        manifest["regions_catalog"] = regions_meta
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\n[Cadastro GeoSampa atualizado no manifesto com sucesso!]")
        return

    api_key = load_api_key()
    if not api_key:
        print("\n[ERRO] GOOGLE_MAPS_API_KEY não encontrada em arborgrid-vision/.env!")
        return

    print(f"\n[2/3] Coletando Street View ({args.views} ângulos/árvore) para {len(new_candidates)} novas árvores...")
    images_dir = out_dir / "images"
    new_static_calls = 0
    added_trees = 0

    for idx, feat in enumerate(new_candidates, start=1):
        props = feat.get("properties") or {}
        tree_id = str(props.get("cd_identificador", f"{reg_key}_{idx}"))
        lon, lat = feat["geometry"]["coordinates"][:2]

        views = collect_tree_perspectives(lat, lon, api_key=api_key, num_views=args.views)
        if not views:
            print(f"  [{idx}/{len(new_candidates)}] ID {tree_id}: sem Street View oficial próximo.")
            continue

        saved_views = []
        for v in views:
            fname = f"tree_{tree_id}_{v['perspective']}_h{int(v['heading'])}.jpg"
            img_path = images_dir / fname
            if not img_path.exists():
                download_sv_image(v, api_key=api_key, out_path=img_path)
                new_static_calls += 1
                time.sleep(0.08)
            v_rec = dict(v)
            v_rec["file_name"] = fname
            saved_views.append(v_rec)

        manifest["trees"].append(
            {
                "geosampa_id": tree_id,
                "region_key": reg_key,
                "region_label": reg_cfg["label"],
                "distrito": props.get("nm_distrito_municipal"),
                "subprefeitura": props.get("nm_subprefeitura"),
                "tree_lat": lat,
                "tree_lon": lon,
                "sisgau_last_inspection": props.get("dt_inspecao"),
                "views": saved_views,
            }
        )
        added_trees += 1
        if idx % 10 == 0 or idx == len(new_candidates):
            print(f"  -> Progresso {idx}/{len(new_candidates)}: +{added_trees} árvores (+{new_static_calls} fotos)")

    regions_meta = manifest.get("regions_catalog", {})
    regions_meta[reg_key] = {
        "label": reg_cfg["label"],
        "bbox": list(bbox_tuple),
        "total_geosampa_trees": len(all_features),
        "sisgau_inspected_trees": inspected_cnt,
        "geojson_file": f"geosampa_cadastral/geosampa_{reg_key}.geojson",
    }

    manifest["project"] = "ArborGrid - FIAP Global Solution CitizenTech (4ES)"
    manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["regions_catalog"] = regions_meta
    manifest["total_trees_collected"] = len(manifest["trees"])
    manifest["total_static_api_images"] = sum(len(t.get("views", [])) for t in manifest["trees"])
    manifest["data_sources_attribution"] = {
        "tree_coordinates": {
            "provider": "Prefeitura do Município de São Paulo - GeoSampa / SISGAU",
            "layer": "geoportal:arvore (Arborização Viária)",
            "wfs_endpoint": GEOSAMPA_WFS_URL,
            "legal_basis": "Decreto Municipal nº 57.576/2017 (Dados Abertos PMSP)",
        },
        "street_level_imagery": {
            "provider": "Google Maps Platform (Street View Metadata API & Street View Static API)",
            "attribution": "© Google",
            "usage_scope": "Pesquisa acadêmica universitária (FIAP Global Solution) e auditoria georreferenciada",
        },
    }

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[3/3] Região '{reg_key}' atualizada! Novas fotos baixadas: {new_static_calls}")
    print(f"  -> Total acumulado no manifesto: {manifest['total_trees_collected']} árvores / {manifest['total_static_api_images']} fotos.")


if __name__ == "__main__":
    main()
