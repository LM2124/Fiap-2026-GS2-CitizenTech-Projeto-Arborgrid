"""
Configuração centralizada do módulo ArborGrid Vision.

Reúne em um único lugar os caminhos de pastas e arquivos, endereços das APIs,
cores visuais e limiares geométricos usados pelos scripts e notebooks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# ── Diretórios principais do repositório ─────────────────────────────────────
SCRIPTS_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPTS_DIR.parent                  # pasta arborgrid-vision/
PROJECT_ROOT = BASE_DIR.parent                 # raiz do projeto ArborGrid/

ASSETS_DIR = BASE_DIR / "assets"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
DATASET_DIR = BASE_DIR / "dataset"

# ── Dados territoriais (GeoSampa WFS + Street View) e WebApp ─────────────────
SP_GEOSAMPA_DIR = DATASET_DIR / "sp_geosampa_api"
SP_IMAGES_RAW_DIR = SP_GEOSAMPA_DIR / "images"
SP_V3_OVERLAYS_DIR = SP_GEOSAMPA_DIR / "v3_overlays"
SP_PRELABELS_YOLO_DIR = SP_GEOSAMPA_DIR / "prelabels_yolo_seg"

REGIONS_PATH = SP_GEOSAMPA_DIR / "regions.json"
MANIFEST_PATH = SP_GEOSAMPA_DIR / "manifest.json"
WEBAPP_MANIFEST_PATH = PROJECT_ROOT / "webapp" / "data" / "manifest.json"

# ── Pacotes de anotação manual (X-AnyLabeling) e dataset consolidado ─────────
XANY_WORKSPACE_DIR = DATASET_DIR / "workspace_xanylabeling"
PACOTE_1_PRELABEL_DIR = XANY_WORKSPACE_DIR / "pacote_1_prelabel"
PACOTE_1_DIR = XANY_WORKSPACE_DIR / "pacote_1"
PACOTE_2_DIR = XANY_WORKSPACE_DIR / "pacote_2"
PACOTE_3_DIR = XANY_WORKSPACE_DIR / "pacote_3_hard_negatives"

YOLO_EXPORT_DIR = DATASET_DIR / "yolo_seg_ready"
GOLD_DATASET_ZIP_PATH = DATASET_DIR / "arborgrid_v3_gold_dataset.zip"

# ── Modelos e pesos treinados (PyTorch .pt e ONNX INT8 .onnx) ────────────────
SEGFORMER_BASE_MODEL = "nvidia/segformer-b2-finetuned-ade-512-512"
PT_PILOT_640_PATH = MODELS_DIR / "arborgrid_segformer_b2_pilot.pt"
PT_V1_1024_PATH = MODELS_DIR / "arborgrid_segformer_b2_1024_gold.pt"
PT_V2_1024_PATH = MODELS_DIR / "arborgrid_segformer_b2_1024_v2_full.pt"
PT_V3_1024_PATH = MODELS_DIR / "arborgrid_segformer_b2_1024_v3_gold.pt"
ONNX_V3_INT8_PATH = MODELS_DIR / "segformer_v3_int8_dyn_matmul.onnx"

# ── APIs externas de coleta em campo ─────────────────────────────────────────
GEOSAMPA_WFS_URL = "https://wfs.geosampa.prefeitura.sp.gov.br/geoserver/geoportal/wfs"
SV_METADATA_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"
SV_IMAGE_URL = "https://maps.googleapis.com/maps/api/streetview"

# ── Aparência visual, tipografia e normalização de entrada ───────────────────
COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "arvore": (40, 210, 60),   # Verde usado para copas de árvore (formato BGR do OpenCV)
    "fiacao": (30, 45, 250),   # Vermelho usado para fiação elétrica (formato BGR do OpenCV)
}

STATUS_COLORS_HEX: dict[str, str] = {
    "CRITICO": "#ef4444",
    "PROXIMIDADE": "#f59e0b",
    "SEGURO": "#10b981",
    "CRITICO_CONFIRMADO": "#ef4444",
    "CRITICO_MONO_VISADA": "#ef4444",
    "DIVERGENCIA_PARALAXE": "#f97316",
    "PROXIMIDADE_MONO_VISADA": "#f59e0b",
    "SEGURO_MONO_VISADA": "#10b981",
}

# Normalização padrão esperada pelo SegFormer (média e desvio do conjunto ImageNet)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Fontes com suporte completo a acentuação em português (Windows e Linux)
TTF_FONTS_REGULAR = [
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
TTF_FONTS_BOLD = [
    "C:/Windows/Fonts/segoeuib.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]

# ── Limiares calibrados do motor geométrico V3 ───────────────────────────────
THRESHOLDS: dict[str, int] = {
    "min_wire_pixels": 120,         # Ignora ruídos minúsculos antes de acusar presença de rede
    "r1_contact_radius_px": 6,      # Raio em volta da copa que indica contato direto com o fio
    "r2_proximity_radius_px": 24,   # Faixa externa ao redor da copa (zona de aproximação pelo vento)
    "critical_contact_px": 35,      # Quantidade mínima de pixels de fio em contato para marcar Crítico
    "attention_contact_px": 8,      # Contato leve que já exige atenção
    "attention_proximity_px": 90,   # Volume de fios na zona externa para marcar Proximidade
}


def geosampa_disponivel() -> bool:
    """Confere se o WFS do GeoSampa responde antes de uma coleta longa."""
    import requests

    resp = requests.get("http://wfs.geosampa.prefeitura.sp.gov.br/geoserver/ows?service=WFS&request=GetCapabilities", timeout=10)
    return resp.ok
