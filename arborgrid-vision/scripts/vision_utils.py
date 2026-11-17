"""
Utilitários compartilhados do ecossistema ArborGrid Vision (Pesquisa & Operação).

Centraliza:
  1. Carregamento e inferência do SegFormer-B2 (1024x1024) via PyTorch (.pt) ou ONNX Runtime (.onnx INT8).
  2. Conversores geométricos de máscaras (X-AnyLabeling JSON e YOLO-Seg TXT -> máscaras binárias).
  3. Motor morfológico V3 de detecção de conflito (r1 = 6 px / r2 = 24 px) e consenso de paralaxe multi-ângulo.
  4. Renderizadores visuais com suporte nativo a UTF-8 (acentuação em português via Pillow TrueType).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── Constantes de Classes, Cores e Limiares Calibrados V3 ────────────────────
COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "arvore": (40, 210, 60),    # Verde vibrante (BGR)
    "fiacao": (30, 45, 250),    # Vermelho vibrante (BGR)
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
    "CRITICAL_INTERSECTION": "#ef4444",
    "ATTENTION_PROXIMITY": "#f97316",
    "SAFE_CLEAR": "#22c55e",
    "SAFE_NO_WIRES": "#3b82f6",
}

THRESHOLDS: dict[str, int] = {
    "min_wire_pixels": 120,
    "min_tree_pixels": 600,
    "r1_contact_radius_px": 6,        # ~1% da largura 640px (contato/oclusão direta)
    "r2_proximity_radius_px": 24,     # ~3.75% da largura 640px (zona de proximidade)
    "critical_contact_px": 35,        # Mínimo de pixels de fiação em contato direto na visão
    "attention_contact_px": 8,        # Contato tênue
    "attention_proximity_px": 90,     # Fiação abundante na zona de guarda
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── Helpers de Tipografia UTF-8 (Pillow TrueType sobre matriz BGR OpenCV) ────
def _load_ttf_font(size: int = 16, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Carrega fonte TrueType com suporte completo a UTF-8 (Windows/Linux fallback)."""
    candidates = (
        [
            "C:/Windows/Fonts/segoeuib.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
        if bold
        else [
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    )
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def draw_utf8_text(
    img_bgr: np.ndarray,
    lines: list[tuple[int, int, str, tuple[int, int, int], int, bool]],
) -> np.ndarray:
    """
    Desenha múltiplas strings UTF-8 em uma imagem BGR usando Pillow.
    Cada item em `lines`: (x, y, text, color_bgr, font_size, bold)
    """
    pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    for x, y, text, color_bgr, font_size, is_bold in lines:
        font = _load_ttf_font(size=font_size, bold=is_bold)
        color_rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
        draw.text((x, y), text, font=font, fill=color_rgb)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def stamp_header(
    img_bgr: np.ndarray,
    title: str,
    subtitle: str = "",
    badge_color: tuple[int, int, int] = (40, 210, 60),
) -> np.ndarray:
    """Adiciona cabeçalho escuro com faixa lateral colorida e texto UTF-8 acentuado."""
    h, w = img_bgr.shape[:2]
    bar_h = 60 if subtitle else 40
    bar = np.full((bar_h, w, 3), (18, 23, 34), dtype=np.uint8)
    cv2.rectangle(bar, (0, 0), (8, bar_h), badge_color, -1)

    text_Specs = [(18, 7, title, (248, 250, 252), 17, True)]
    if subtitle:
        text_Specs.append((18, 33, subtitle, (165, 180, 202), 13, False))
    bar = draw_utf8_text(bar, text_Specs)
    return np.vstack([bar, img_bgr])


def add_legend_bar(
    img_bgr: np.ndarray,
    items: list[tuple[tuple[int, int, int], str]],
    footer_note: str = "",
) -> np.ndarray:
    """Adiciona barra de legenda inferior com caixas coloridas e texto UTF-8."""
    w = img_bgr.shape[1]
    bar_h = 44
    bar = np.full((bar_h, w, 3), (15, 23, 42), dtype=np.uint8)

    pil_img = Image.fromarray(cv2.cvtColor(bar, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font_bold = _load_ttf_font(size=14, bold=True)
    font_note = _load_ttf_font(size=13, bold=False)

    x = 18
    for color_bgr, text in items:
        color_rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
        draw.rectangle([x, 13, x + 22, 31], fill=color_rgb, outline=(226, 232, 240))
        draw.text((x + 30, 12), text, font=font_bold, fill=(241, 245, 249))
        bbox = draw.textbbox((0, 0), text, font=font_bold)
        text_w = bbox[2] - bbox[0]
        x += 30 + text_w + 28

    if footer_note:
        bbox = draw.textbbox((0, 0), footer_note, font=font_note)
        note_w = bbox[2] - bbox[0]
        draw.text(
            (max(x, w - note_w - 18), 13),
            footer_note,
            font=font_note,
            fill=(148, 163, 184),
        )

    bar_out = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return np.vstack([img_bgr, bar_out])


# ── Mapeamento de Caminhos e Conversores de Anotação ─────────────────────────
def remap_colab_to_local(colab_path: str, base_dir: Path) -> Path:
    """Converte caminhos de treino do Google Colab para a árvore local `dataset/workspace_xanylabeling/`."""
    ws = base_dir / "dataset" / "workspace_xanylabeling"
    p = colab_path.replace("\\", "/")
    if "pacote_colega_1_labelled/" in p:
        rel = p.split("pacote_colega_1_labelled/", 1)[1]
        return ws / "pacote_1" / Path(rel).name
    if "pacote_1/" in p:
        rel = p.split("pacote_1/", 1)[1]
        return ws / "pacote_1" / Path(rel).name
    if "pacote_colega_2/" in p:
        rel = p.split("pacote_colega_2/", 1)[1]
        return ws / "pacote_2" / Path(rel).name
    if "pacote_2/" in p:
        rel = p.split("pacote_2/", 1)[1]
        return ws / "pacote_2" / Path(rel).name
    if "pacote_3_hard_negatives/" in p:
        rel = p.split("pacote_3_hard_negatives/", 1)[1]
        return ws / "pacote_3_hard_negatives" / Path(rel).name
    return Path(colab_path)


def ensure_min_wire_thickness(mask_wire: np.ndarray, min_px: int = 3) -> np.ndarray:
    """Garante espessura mínima (3px) em cabos elétricos para evitar perda por sub-amostragem."""
    if mask_wire.sum() == 0:
        return mask_wire
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (min_px, min_px))
    return cv2.dilate(mask_wire, kernel, iterations=1)


def json_to_masks(
    json_path: Path, h: int = 640, w: int = 640
) -> tuple[np.ndarray, np.ndarray]:
    """Converte arquivo JSON do X-AnyLabeling em máscaras binárias `(mask_tree, mask_wire)`."""
    mask_tree = np.zeros((h, w), dtype=np.uint8)
    mask_wire = np.zeros((h, w), dtype=np.uint8)
    if not json_path.exists():
        return mask_tree, mask_wire
    data = json.loads(json_path.read_text(encoding="utf-8"))
    orig_w = float(data.get("imageWidth") or w)
    orig_h = float(data.get("imageHeight") or h)
    sx = w / max(orig_w, 1.0)
    sy = h / max(orig_h, 1.0)
    for shape in data.get("shapes", []):
        label = shape.get("label", "").lower().strip()
        pts_raw = shape.get("points", [])
        if len(pts_raw) < 3:
            continue
        pts = np.round(np.array([[p[0] * sx, p[1] * sy] for p in pts_raw])).astype(np.int32)
        if "copa" in label or label in ("arvore", "árvore", "tree", "vegetacao"):
            cv2.fillPoly(mask_tree, [pts], 1)
        elif "fio" in label or "fiacao" in label or "fiação" in label or label in ("wire", "cable", "rede_eletrica"):
            cv2.fillPoly(mask_wire, [pts], 1)
    return mask_tree, mask_wire


def yolo_txt_to_masks(
    txt_path: Path, h: int = 640, w: int = 640
) -> tuple[np.ndarray, np.ndarray]:
    """Converte arquivo `.txt` YOLO-Seg em máscaras binárias `(mask_tree, mask_wire)`."""
    mask_tree = np.zeros((h, w), dtype=np.uint8)
    mask_wire = np.zeros((h, w), dtype=np.uint8)
    if not txt_path.exists():
        return mask_tree, mask_wire
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        cls_id = int(parts[0])
        coords = np.array([float(x) for x in parts[1:]], dtype=np.float32).reshape(-1, 2)
        pts = np.round(coords * np.array([w, h], dtype=np.float32)).astype(np.int32)
        if cls_id == 0:
            cv2.fillPoly(mask_tree, [pts], 1)
        elif cls_id == 1:
            cv2.fillPoly(mask_wire, [pts], 1)
    return mask_tree, mask_wire


# ── Renderizadores de Overlays e Painéis de Discrepância ─────────────────────
def render_overlay(
    img_bgr: np.ndarray,
    mask_tree: np.ndarray,
    mask_wire: np.ndarray,
    alpha_tree: float = 0.36,
    alpha_wire: float = 0.25,
    wire_contour_px: int = 1,
) -> np.ndarray:
    """
    Renderiza sobreposição semitransparente de árvore (verde) e fiação (vermelho)
    com contorno externo fino de 1px para preservar a visibilidade dos cabos reais.
    """
    canvas = img_bgr.copy()

    if mask_tree.sum() > 0:
        b = mask_tree > 0
        color_layer = np.zeros_like(canvas)
        color_layer[:] = COLORS_BGR["arvore"]
        canvas[b] = cv2.addWeighted(canvas, 1.0 - alpha_tree, color_layer, alpha_tree, 0)[b]
        contours, _ = cv2.findContours(mask_tree, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, (20, 140, 35), 1, cv2.LINE_AA)

    if mask_wire.sum() > 0:
        b = mask_wire > 0
        color_layer = np.zeros_like(canvas)
        color_layer[:] = COLORS_BGR["fiacao"]
        canvas[b] = cv2.addWeighted(canvas, 1.0 - alpha_wire, color_layer, alpha_wire, 0)[b]
        if wire_contour_px > 0:
            contours, _ = cv2.findContours(mask_wire, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, contours, -1, (80, 100, 255), wire_contour_px, cv2.LINE_AA)

    return canvas


def render_diff_panel(
    img_bgr: np.ndarray,
    gt_t: np.ndarray | None,
    gt_w: np.ndarray,
    pr_t: np.ndarray | None,
    pr_w: np.ndarray,
    alpha: float = 0.58,
    wire_only: bool = True,
) -> np.ndarray:
    """
    Painel de discrepância pixel a pixel com transparência (`alpha`) sobre a foto:
      - Verde: Previsão correta / Acordo (Interseção GT ∩ Pred)
      - Vermelho: Falso Positivo / Extra previsto pelo modelo sem estar no GT
      - Amarelo: Região perdida / Falso Negativo (anotado no GT, omitido pelo modelo)
    Por padrão (`wire_only=True`), foca exclusivamente na classe Fiação.
    """
    canvas = (img_bgr.astype(np.float32) * 0.72).astype(np.uint8)
    overlay = canvas.copy()

    if wire_only or gt_t is None or pr_t is None:
        gt_mask = gt_w > 0
        pr_mask = pr_w > 0
    else:
        gt_mask = (gt_t > 0) | (gt_w > 0)
        pr_mask = (pr_t > 0) | (pr_w > 0)

    tp = gt_mask & pr_mask
    fp = pr_mask & (~gt_mask)
    fn = gt_mask & (~pr_mask)

    overlay[tp] = (40, 215, 70)   # Verde (BGR)
    overlay[fp] = (40, 45, 250)   # Vermelho (BGR)
    overlay[fn] = (0, 225, 255)   # Amarelo (BGR)

    mask_any = tp | fp | fn
    if np.any(mask_any):
        blended = cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0)
        canvas[mask_any] = blended[mask_any]
        for region_mask, edge_color in [
            (tp, (25, 160, 45)),
            (fp, (30, 35, 210)),
            (fn, (0, 175, 215)),
        ]:
            cnts, _ = cv2.findContours(
                region_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if cnts:
                cv2.drawContours(canvas, cnts, -1, edge_color, 1, cv2.LINE_AA)
    return canvas


def draw_morphology_panel(
    img_bgr: np.ndarray,
    mask_tree: np.ndarray,
    mask_wire: np.ndarray,
) -> tuple[np.ndarray, int, int, str]:
    """
    Renderiza o painel geométrico da Morfologia Matemática V3:
      - Contorno amarelo tracejado/anel externo: Zona de Proximidade (r2 = 24 px)
      - Contorno laranja interno: Zona de Contato Direto (r1 = 6 px)
      - Pixels em vermelho vivo/ciano: Fiação em contato direto / buffer de vento
    Retorna `(canvas_bgr, contact_px, buffer_px, status_label_utf8)`.
    """
    canvas = (img_bgr.astype(np.float32) * 0.48).astype(np.uint8)
    canvas = render_overlay(
        canvas, mask_tree, mask_wire, alpha_tree=0.32, alpha_wire=0.22, wire_contour_px=1
    )

    k_r1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))  # r1 = 6px -> 2*6+1 = 13
    k_r2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (49, 49))  # r2 = 24px -> 2*24+1 = 49
    dil_r1 = cv2.dilate(mask_tree, k_r1)
    dil_r2 = cv2.dilate(mask_tree, k_r2)

    cnt_r2, _ = cv2.findContours(dil_r2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, cnt_r2, -1, (0, 215, 255), 1, cv2.LINE_AA)

    cnt_r1, _ = cv2.findContours(dil_r1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, cnt_r1, -1, (0, 140, 255), 1, cv2.LINE_AA)

    contact_direct = (dil_r1 > 0) & (mask_wire > 0)
    buffer_zone = (dil_r2 > 0) & (dil_r1 == 0) & (mask_wire > 0)

    canvas[buffer_zone] = (0, 230, 255)   # Amarelo brilhante nos fios em zona de proximidade
    canvas[contact_direct] = (255, 255, 0)  # Ciano brilhante nos fios em contato direto

    c_px = int(contact_direct.sum())
    b_px = int(buffer_zone.sum())
    w_px = int(mask_wire.sum())

    if w_px < 120:
        status = "SEGURO (SEM FIAÇÃO)"
    elif c_px >= 35:
        status = "CRÍTICO (CONTATO DIRETO)"
    elif c_px >= 8 or b_px >= 90:
        status = "PROXIMIDADE (ALERTA)"
    else:
        status = "SEGURO (LIVRE)"

    return canvas, c_px, b_px, status


# ── Carregamento e Inferência SegFormer-B2 (PyTorch .pt & ONNX Runtime) ──────
def _remap_segformer_state_dict(sd: dict[str, Any]) -> dict[str, Any]:
    """Converte chaves de checkpoints SegFormer salvos no Colab (transformers 5.x) para local (transformers 4.x)."""
    import re

    out = {}
    for k, v in sd.items():
        nk = k
        nk = re.sub(
            r"^segformer\.stages\.(\d+)\.patch_embeddings\.",
            r"segformer.encoder.patch_embeddings.\1.",
            nk,
        )
        nk = re.sub(
            r"^segformer\.stages\.(\d+)\.layer_norm\.",
            r"segformer.encoder.layer_norm.\1.",
            nk,
        )
        nk = re.sub(
            r"^segformer\.stages\.(\d+)\.blocks\.(\d+)\.",
            r"segformer.encoder.block.\1.\2.",
            nk,
        )
        nk = nk.replace(".layernorm_before.", ".layer_norm_1.")
        nk = nk.replace(".layernorm_after.", ".layer_norm_2.")
        nk = nk.replace(".attention.q_proj.", ".attention.self.query.")
        nk = nk.replace(".attention.k_proj.", ".attention.self.key.")
        nk = nk.replace(".attention.v_proj.", ".attention.self.value.")
        nk = nk.replace(".attention.o_proj.", ".attention.output.dense.")
        nk = nk.replace(
            ".attention.sequence_reduction.sequence_reduction.",
            ".attention.self.sr.",
        )
        nk = nk.replace(
            ".attention.sequence_reduction.layer_norm.",
            ".attention.self.layer_norm.",
        )
        nk = nk.replace(".mlp.fc1.", ".mlp.dense1.")
        nk = nk.replace(".mlp.fc2.", ".mlp.dense2.")
        nk = re.sub(
            r"^decode_head\.linear_projections\.(\d+)\.",
            r"decode_head.linear_c.\1.",
            nk,
        )
        out[nk] = v
    return out


def load_segformer_weights(ckpt_path: Path, device: Any) -> Any:
    """Carrega checkpoint `.pt` do SegFormer-B2 (3 classes: bg, arvore, fiacao) com remapeamento Colab->Local."""
    import torch
    from transformers import SegformerForSemanticSegmentation

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_name = (
        ckpt.get("model_name", "nvidia/segformer-b2-finetuned-ade-512-512")
        if isinstance(ckpt, dict) and "model_name" in ckpt
        else "nvidia/segformer-b2-finetuned-ade-512-512"
    )
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_name, num_labels=3, ignore_mismatched_sizes=True
    ).to(device)
    state_dict = (
        ckpt["model_state_dict"]
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt
        else (ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt)
    )
    if any(k.startswith("segformer.stages.") for k in state_dict):
        state_dict = _remap_segformer_state_dict(state_dict)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_onnx_session(onnx_path: Path) -> Any:
    """Inicializa sessão ONNX Runtime CPU para `segformer_v3_int8_dyn_matmul.onnx`."""
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(onnx_path), sess_options=sess_options, providers=["CPUExecutionProvider"]
    )


def _preprocess_1024(img_bgr: np.ndarray, res: int = 1024) -> np.ndarray:
    """Converte imagem BGR para tensor NCHW float32 normalizado (ImageNet) em `res x res`."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (res, res), interpolation=cv2.INTER_LINEAR)
    norm = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return norm.transpose(2, 0, 1)[np.newaxis, ...]  # (1, 3, res, res)


def _sigmoid_numpy(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))


def infer_probs(
    model_or_session: Any,
    img_bgr: np.ndarray,
    device: Any = None,
    res: int = 1024,
) -> np.ndarray:
    """
    Executa forward pass em 1024x1024 (PyTorch Module ou ONNX InferenceSession)
    e retorna matriz `(3, H, W)` de probabilidades Sigmoid (`0: copa, 1: fiacao, 2: poste`)
    na resolução original da imagem.
    """
    orig_h, orig_w = img_bgr.shape[:2]
    inp_np = _preprocess_1024(img_bgr, res=res)

    # Caso 1: Sessão ONNX Runtime
    if hasattr(model_or_session, "get_inputs"):
        input_name = model_or_session.get_inputs()[0].name
        logits = model_or_session.run(None, {input_name: inp_np})[0]  # (1, 3, 256, 256)
        probs_low = _sigmoid_numpy(logits)[0]  # (3, 256, 256)
        probs = np.stack(
            [
                cv2.resize(probs_low[c], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                for c in range(probs_low.shape[0])
            ],
            axis=0,
        )
        return probs

    # Caso 2: Modelo PyTorch
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        inp = torch.from_numpy(inp_np).to(device)
        use_amp = device is not None and getattr(device, "type", "cpu") == "cuda"
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model_or_session(pixel_values=inp).logits
            up = F.interpolate(
                logits.float(), size=(orig_h, orig_w), mode="bilinear", align_corners=False
            )
            probs = torch.sigmoid(up)[0].cpu().numpy()
    return probs


def infer_masks(
    model_or_session: Any,
    img_bgr: np.ndarray,
    device: Any = None,
    res: int = 1024,
    tree_thresh: float = 0.50,
    wire_thresh: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Retorna `(mask_tree, mask_wire)` binárias (uint8 0/1) onde canal 0=copa e canal 1=fiacao."""
    probs = infer_probs(model_or_session, img_bgr, device=device, res=res)
    w_th = 0.35 if wire_thresh is None else wire_thresh
    mask_tree = (probs[0] >= tree_thresh).astype(np.uint8)
    mask_wire = (probs[1] >= w_th).astype(np.uint8)
    return mask_tree, mask_wire


# ── Motor Morfológico V3 (r1=6px / r2=24px) e Consenso de Paralaxe ───────────
def analyze_conflict_morphology_calibrated(
    mask_tree: np.ndarray,
    mask_wire: np.ndarray,
    prob_wire: np.ndarray,
) -> dict[str, Any]:
    """
    Calcula a interação espacial 2D entre copa arbórea (`mask_tree`) e rede aérea (`mask_wire`)
    utilizando dois anéis morfológicos concêntricos:
      - Zona de Contato Direto (r1 = 6 px): Interseção/oclusão visual direta.
      - Zona de Guarda / Proximidade (r2 = 24 px): Corredor de alerta preventivo.
    """
    tree_px = int(mask_tree.sum())
    wire_px = int(mask_wire.sum())

    if wire_px < THRESHOLDS["min_wire_pixels"] or tree_px < THRESHOLDS["min_tree_pixels"]:
        return {
            "status": "SAFE_NO_WIRES" if wire_px < THRESHOLDS["min_wire_pixels"] else "SAFE_CLEAR",
            "severity_score": 0.0,
            "tree_pixels": tree_px,
            "wire_pixels": wire_px,
            "contact_pixels": 0,
            "proximity_pixels": 0,
            "wire_in_tree_Envelope_ratio": 0.0,
            "mean_wire_confidence": round(
                float(prob_wire[mask_wire == 1].mean()) if wire_px > 0 else 0.0, 4
            ),
        }

    r1 = THRESHOLDS["r1_contact_radius_px"]
    r2 = THRESHOLDS["r2_proximity_radius_px"]
    k_contact = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r1 + 1, 2 * r1 + 1))
    k_prox = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r2 + 1, 2 * r2 + 1))

    tree_contact_zone = cv2.dilate(mask_tree, k_contact)
    tree_prox_zone = cv2.dilate(mask_tree, k_prox)

    contact_mask = (tree_contact_zone == 1) & (mask_wire == 1)
    prox_mask = (tree_prox_zone == 1) & (mask_wire == 1) & (~contact_mask)

    contact_px = int(contact_mask.sum())
    prox_px = int(prox_mask.sum())
    wire_conf = float(prob_wire[mask_wire == 1].mean()) if wire_px > 0 else 0.0
    envelope_ratio = float((contact_px + prox_px) / max(wire_px, 1))

    if contact_px >= THRESHOLDS["critical_contact_px"] or (
        contact_px >= 20 and envelope_ratio >= 0.28
    ):
        status = "CRITICAL_INTERSECTION"
        raw_sev = 0.65 + min(0.35, (contact_px / 220.0) * 0.25 + envelope_ratio * 0.15)
    elif (
        contact_px >= THRESHOLDS["attention_contact_px"]
        or prox_px >= THRESHOLDS["attention_proximity_px"]
    ):
        status = "ATTENTION_PROXIMITY"
        raw_sev = 0.30 + min(0.34, (contact_px / 40.0) * 0.15 + (prox_px / 350.0) * 0.20)
    else:
        status = "SAFE_CLEAR"
        raw_sev = min(0.25, (prox_px / 100.0) * 0.20)

    return {
        "status": status,
        "severity_score": round(float(raw_sev * (0.75 + 0.25 * wire_conf)), 3),
        "tree_pixels": tree_px,
        "wire_pixels": wire_px,
        "contact_pixels": contact_px,
        "proximity_pixels": prox_px,
        "wire_in_tree_Envelope_ratio": round(envelope_ratio, 4),
        "mean_wire_confidence": round(wire_conf, 4),
    }


def consolidate_tree_triage(views_results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Regra de Consenso Multi-Ângulo (Mitigação de Falso-Positivo por Paralaxe 2D):
    Uma árvore só recebe `CRITICAL_INTERSECTION` se:
      1. Pelo menos 2 ângulos independentes apresentarem risco (CRITICAL ou ATTENTION), OU
      2. Um único ângulo crítico apresentar contato maciço (>= 95 px) com alta razão de envelope (>= 0.35).
    Caso contrário, um único ângulo crítico isolado é rebaixado para `ATTENTION_PROXIMITY`.
    """
    priority = {
        "CRITICAL_INTERSECTION": 4,
        "ATTENTION_PROXIMITY": 3,
        "SAFE_CLEAR": 2,
        "SAFE_NO_WIRES": 1,
    }
    valid_views = [v for v in views_results if "inference" in v]
    if not valid_views:
        return {
            "overall_status": "PENDING_INFERENCE",
            "max_severity_score": 0.0,
            "color_hex": "#9ca3af",
            "critical_views": 0,
            "attention_views": 0,
            "parallax_mitigated": False,
        }

    crit_views = [
        v for v in valid_views if v["inference"]["status"] == "CRITICAL_INTERSECTION"
    ]
    att_views = [
        v for v in valid_views if v["inference"]["status"] == "ATTENTION_PROXIMITY"
    ]
    best = max(
        valid_views,
        key=lambda v: (
            priority.get(v["inference"]["status"], 0),
            v["inference"]["severity_score"],
        ),
    )

    overall_status = best["inference"]["status"]
    max_sev = float(best["inference"]["severity_score"])
    parallax_mitigated = False

    if len(crit_views) == 1 and len(att_views) == 0 and len(valid_views) >= 2:
        c_inf = crit_views[0]["inference"]
        if (
            c_inf["contact_pixels"] < 95
            or c_inf.get("wire_in_tree_Envelope_ratio", 0.0) < 0.35
        ):
            overall_status = "ATTENTION_PROXIMITY"
            max_sev = min(max_sev, 0.58)
            parallax_mitigated = True

    return {
        "overall_status": overall_status,
        "max_severity_score": round(max_sev, 3),
        "dominant_view": best.get("view_type", "center_front"),
        "color_hex": STATUS_COLORS_HEX.get(overall_status, "#9ca3af"),
        "critical_views": len(crit_views),
        "attention_views": len(att_views),
        "parallax_mitigated": parallax_mitigated,
    }


def analyze_view_v3_manifest(
    mask_tree: np.ndarray,
    mask_wire: np.ndarray,
    file_name: str,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """
    Calcula o dicionário `v3_analysis` compatível com `manifest.json` e `webapp/data/manifest.json`
    a partir das máscaras binárias `mask_tree` e `mask_wire` (com r1 = 6 px e r2 = 24 px).
    Retorna `(v3_analysis_dict, contact_mask, buffer_mask)`.
    """
    copa_px = int(mask_tree.sum())
    wire_px = int(mask_wire.sum())

    r1 = THRESHOLDS["r1_contact_radius_px"]
    r2 = THRESHOLDS["r2_proximity_radius_px"]
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r1 + 1, 2 * r1 + 1))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r2 + 1, 2 * r2 + 1))

    dil_r1 = cv2.dilate(mask_tree, k1)
    dil_r2 = cv2.dilate(mask_tree, k2)

    contact_mask = (dil_r1 == 1) & (mask_wire == 1)
    buffer_mask = (dil_r2 == 1) & (mask_wire == 1) & (~contact_mask)

    contact_px = int(contact_mask.sum())
    buffer_px = int(buffer_mask.sum())

    if wire_px < THRESHOLDS["min_wire_pixels"]:
        triage = "SEGURO"
    elif contact_px >= THRESHOLDS["critical_contact_px"]:
        triage = "CRITICO"
    elif (
        contact_px >= THRESHOLDS["attention_contact_px"]
        or buffer_px >= THRESHOLDS["attention_proximity_px"]
    ):
        triage = "PROXIMIDADE"
    else:
        triage = "SEGURO"

    return (
        {
            "triage": triage,
            "color_hex": STATUS_COLORS_HEX[triage],
            "copa_px": copa_px,
            "wire_px": wire_px,
            "pole_px": 0,
            "contact_direct_px": contact_px,
            "wind_buffer_px": buffer_px,
            "v3_overlay_file": f"ov_v3_{file_name}",
        },
        contact_mask.astype(np.uint8),
        buffer_mask.astype(np.uint8),
    )


def consolidate_tree_v3_manifest(views: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Consolida a triagem multi-ângulo no formato `v3_triage` do `manifest.json`
    aplicando a Regra de Consenso de Paralaxe (`CRITICO_CONFIRMADO`, `DIVERGENCIA_PARALAXE`, `PROXIMIDADE`, `SEGURO`).
    """
    v_analyses = [v.get("v3_analysis", {}) for v in views if "v3_analysis" in v]
    if not v_analyses:
        return {
            "front_triage": "SEGURO",
            "front_color": "#10b981",
            "side_triage": "SEGURO",
            "side_color": "#10b981",
            "consolidated": "SEGURO",
            "has_paralax_divergence": False,
            "has_single_view": len(views) == 1,
        }

    front = v_analyses[0]
    side = v_analyses[1] if len(v_analyses) > 1 else front
    f_tri = front.get("triage", "SEGURO")
    s_tri = side.get("triage", "SEGURO")

    has_single = len(v_analyses) == 1
    has_div = False

    if has_single:
        consolidated = f"{f_tri}_MONO_VISADA"
    elif f_tri == "CRITICO" and s_tri == "CRITICO":
        consolidated = "CRITICO_CONFIRMADO"
    elif "CRITICO" in (f_tri, s_tri):
        other_tri = s_tri if f_tri == "CRITICO" else f_tri
        if other_tri == "PROXIMIDADE":
            consolidated = "CRITICO_CONFIRMADO"
        else:
            consolidated = "DIVERGENCIA_PARALAXE"
            has_div = True
    elif "PROXIMIDADE" in (f_tri, s_tri):
        consolidated = "PROXIMIDADE"
    else:
        consolidated = "SEGURO"

    return {
        "front_triage": f_tri,
        "front_color": STATUS_COLORS_HEX.get(f_tri, "#10b981"),
        "side_triage": "UNICA_VISAO" if has_single else s_tri,
        "side_color": "#64748b" if has_single else STATUS_COLORS_HEX.get(s_tri, "#10b981"),
        "consolidated": consolidated,
        "has_paralax_divergence": has_div,
        "has_single_view": has_single,
    }

