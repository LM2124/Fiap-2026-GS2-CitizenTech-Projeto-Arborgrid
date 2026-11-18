"""
Funções compartilhadas do ArborGrid Vision (pesquisa nos notebooks e pipeline operacional).

Reúne:
  1. Desenho de textos e legendas com acentuação (via Pillow TrueType, já que o OpenCV só aceita ASCII).
  2. Conversão das marcações manuais (JSON do X-AnyLabeling e TXT do YOLO) em máscaras binárias.
  3. Overlays visuais (máscaras normais, mapas de diferença focados em fiação e anéis morfológicos).
  4. Carregamento e execução do SegFormer-B2 em PyTorch (.pt) ou ONNX Runtime (.onnx INT8).
  5. Regras geométricas V3 de distância árvore–fio e consenso entre dois ângulos da mesma árvore.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import (
    COLORS_BGR,
    IMAGENET_MEAN,
    IMAGENET_STD,
    PACOTE_1_DIR,
    PACOTE_1_PRELABEL_DIR,
    PACOTE_2_DIR,
    PACOTE_3_DIR,
    SEGFORMER_BASE_MODEL,
    STATUS_COLORS_HEX,
    THRESHOLDS,
    TTF_FONTS_BOLD,
    TTF_FONTS_REGULAR,
)


# ── Escrita de texto em português sobre imagens do OpenCV ────────────────────
def _load_ttf_font(
    size: int = 16, bold: bool = False
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Abre uma fonte TrueType do sistema para permitir acentos e cedilha nas legendas."""
    candidates = TTF_FONTS_BOLD if bold else TTF_FONTS_REGULAR
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
    """Desenha múltiplas linhas com suporte a acentuação em uma imagem no formato BGR."""
    # Converte temporariamente de BGR (OpenCV) para RGB (Pillow) antes de escrever
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
    """Adiciona uma faixa escura no topo da imagem com marca lateral colorida e título."""
    w = img_bgr.shape[1]
    bar_h = 60 if subtitle else 40
    bar = np.full((bar_h, w, 3), (18, 23, 34), dtype=np.uint8)
    cv2.rectangle(bar, (0, 0), (8, bar_h), badge_color, -1)

    text_specs = [(18, 7, title, (248, 250, 252), 17, True)]
    if subtitle:
        text_specs.append((18, 33, subtitle, (165, 180, 202), 13, False))
    bar = draw_utf8_text(bar, text_specs)
    return np.vstack([bar, img_bgr])


def add_legend_bar(
    img_bgr: np.ndarray,
    items: list[tuple[tuple[int, int, int], str]],
    footer_note: str = "",
) -> np.ndarray:
    """Adiciona rodapé escuro com quadrados de cor, rótulos e nota explicativa à direita."""
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
        x += 30 + (bbox[2] - bbox[0]) + 28

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


# ── Localização de arquivos e leitura das marcações ──────────────────────────
def remap_colab_to_local(colab_path: str) -> Path:
    """Converte caminhos antigos salvos em relatórios do Colab para as pastas locais atuais."""
    p = colab_path.replace("\\", "/")
    folder_map = {
        "pacote_1_prelabel/": PACOTE_1_PRELABEL_DIR,
        "pacote_1/": PACOTE_1_DIR,
        "pacote_2/": PACOTE_2_DIR,
        "pacote_3_hard_negatives/": PACOTE_3_DIR,
    }
    for marker, target_dir in folder_map.items():
        if marker in p:
            rel = p.split(marker, 1)[1]
            return target_dir / Path(rel).name
    return Path(colab_path)


def ensure_min_wire_thickness(mask_wire: np.ndarray, min_px: int = 3) -> np.ndarray:
    """Engrossa levemente fios muito finos para que não sumam quando a imagem é reduzida."""
    if mask_wire.sum() == 0:
        return mask_wire
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (min_px, min_px))
    return cv2.dilate(mask_wire, kernel, iterations=1)


def json_to_masks(
    json_path: Path, h: int = 640, w: int = 640
) -> tuple[np.ndarray, np.ndarray]:
    """Lê os polígonos desenhados no X-AnyLabeling (.json) e gera `(mask_tree, mask_wire)`."""
    mask_tree = np.zeros((h, w), dtype=np.uint8)
    mask_wire = np.zeros((h, w), dtype=np.uint8)
    if not json_path.exists():
        return mask_tree, mask_wire

    data = json.loads(json_path.read_text(encoding="utf-8"))
    # Ajusta as coordenadas caso o tamanho esperado difira do tamanho original salvo no JSON
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
    """Lê polígonos normalizados de um arquivo YOLO-Seg (.txt) e gera `(mask_tree, mask_wire)`."""
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


# ── Geração de painéis visuais (overlays, diferenças e anéis morfológicos) ────
def render_overlay(
    img_bgr: np.ndarray,
    mask_tree: np.ndarray,
    mask_wire: np.ndarray,
    alpha_tree: float = 0.36,
    alpha_wire: float = 0.25,
    wire_contour_px: int = 1,
) -> np.ndarray:
    """Pinta árvore (verde) e fiação (vermelho) com transparência e borda fina de 1 px."""
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
    Mostra onde a previsão acerta ou diverge do gabarito usando overlay semitransparente:
      - Verde: previsão correta (acordo entre gabarito e modelo)
      - Vermelho: falso positivo (modelo marcou sem estar no gabarito)
      - Amarelo: região perdida (estava no gabarito, mas o modelo não marcou)
    """
    # Escurece um pouco a foto original para o overlay colorido ficar bem legível por cima
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
        # Contorno fino ajuda a enxergar fios finos mesmo quando a transparência mistura com galhos
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
    Desenha a zona de aproximação ao redor da copa para ilustrar a medição geométrica de conflito:
      - Contorno amarelo externo (r2 = 24 px): faixa de aproximação (risco com vento)
      - Fios em ciano/amarelo translúcidos: trechos do cabo em contato direto (r1 = 6 px) ou na zona de vento (r2)
    """
    canvas = render_overlay(
        img_bgr, mask_tree, mask_wire, alpha_tree=0.32, alpha_wire=0.22, wire_contour_px=1
    )

    r1 = THRESHOLDS["r1_contact_radius_px"]
    r2 = THRESHOLDS["r2_proximity_radius_px"]
    k_r1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r1 + 1, 2 * r1 + 1))
    k_r2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r2 + 1, 2 * r2 + 1))
    dil_r1 = cv2.dilate(mask_tree, k_r1)
    dil_r2 = cv2.dilate(mask_tree, k_r2)

    cnt_r2, _ = cv2.findContours(dil_r2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, cnt_r2, -1, (0, 215, 255), 1, cv2.LINE_AA)

    contact_direct = (dil_r1 > 0) & (mask_wire > 0)
    buffer_zone = (dil_r2 > 0) & (dil_r1 == 0) & (mask_wire > 0)

    if np.any(buffer_zone):
        canvas[buffer_zone] = (
            0.42 * canvas[buffer_zone] + 0.58 * np.array([0, 230, 255], dtype=np.float32)
        ).astype(np.uint8)  # Amarelo translúcido nos fios dentro do anel externo
    if np.any(contact_direct):
        canvas[contact_direct] = (
            0.38 * canvas[contact_direct] + 0.62 * np.array([255, 255, 0], dtype=np.float32)
        ).astype(np.uint8)  # Ciano translúcido nos fios tocando o anel interno

    c_px = int(contact_direct.sum())
    b_px = int(buffer_zone.sum())
    w_px = int(mask_wire.sum())

    if w_px < THRESHOLDS["min_wire_pixels"]:
        status = "SEGURO (SEM FIAÇÃO)"
    elif c_px >= THRESHOLDS["critical_contact_px"]:
        status = "CRÍTICO (CONTATO DIRETO)"
    elif c_px >= THRESHOLDS["attention_contact_px"] or b_px >= THRESHOLDS["attention_proximity_px"]:
        status = "PROXIMIDADE (ALERTA)"
    else:
        status = "SEGURO (LIVRE)"

    return canvas, c_px, b_px, status


# ── Carregamento e execução do SegFormer-B2 (PyTorch .pt ou ONNX Runtime) ────
def _remap_segformer_state_dict(sd: dict[str, Any]) -> dict[str, Any]:
    """
    Traduz os nomes das camadas dos arquivos `.pt` salvos no Colab para o formato local.

    Por que isso existe: os modelos foram treinados no Google Colab usando a versão 5.x da
    biblioteca `transformers`, enquanto este ambiente local usa a versão 4.x. Os pesos numéricos
    são exatamente os mesmos, mas a biblioteca mudou os nomes internos das camadas entre as duas
    versões. Sem renomear essas chaves antes de carregar, o PyTorch acusa erro de incompatibilidade.
    """
    out = {}
    for k, v in sd.items():
        nk = k

        # 1. Blocos principais do codificador: `stages.N` (v5) -> `encoder` (v4)
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

        # 2. Normalizações e projeções de atenção dentro de cada bloco
        nk = nk.replace(".layernorm_before.", ".layer_norm_1.")
        nk = nk.replace(".layernorm_after.", ".layer_norm_2.")
        nk = nk.replace(".attention.q_proj.", ".attention.self.query.")
        nk = nk.replace(".attention.k_proj.", ".attention.self.key.")
        nk = nk.replace(".attention.v_proj.", ".attention.self.value.")
        nk = nk.replace(".attention.o_proj.", ".attention.output.dense.")

        # 3. Redução espacial da atenção e camadas densas finais de cada bloco
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

        # 4. Cabeçote de saída: `linear_projections` (v5) -> `linear_c` (v4)
        nk = re.sub(
            r"^decode_head\.linear_projections\.(\d+)\.",
            r"decode_head.linear_c.\1.",
            nk,
        )
        out[nk] = v
    return out


def load_segformer_weights(ckpt_path: Path, device: Any) -> Any:
    """Carrega um checkpoint `.pt` do SegFormer-B2 (canais: copa, fiação e poste)."""
    import torch
    from transformers import SegformerForSemanticSegmentation

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_name = (
        ckpt.get("model_name", SEGFORMER_BASE_MODEL)
        if isinstance(ckpt, dict) and "model_name" in ckpt
        else SEGFORMER_BASE_MODEL
    )
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_name, num_labels=3, ignore_mismatched_sizes=True
    ).to(device)
    state_dict = (
        ckpt["model_state_dict"]
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt
        else (ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt)
    )
    # Detecta chaves da v5 do Colab e converte para o formato local da v4 quando necessário
    if any(k.startswith("segformer.stages.") for k in state_dict):
        state_dict = _remap_segformer_state_dict(state_dict)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_onnx_session(onnx_path: Path) -> Any:
    """Abre uma sessão ONNX Runtime em CPU (usada pelo modelo INT8 quantizado)."""
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(onnx_path), sess_options=sess_options, providers=["CPUExecutionProvider"]
    )


def _preprocess_1024(img_bgr: np.ndarray, res: int = 1024) -> np.ndarray:
    """Redimensiona para `res x res`, troca BGR->RGB e normaliza para o formato esperado pela rede."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (res, res), interpolation=cv2.INTER_LINEAR)
    norm = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return norm.transpose(2, 0, 1)[np.newaxis, ...]


def _sigmoid_numpy(logits: np.ndarray) -> np.ndarray:
    """Converte valores brutos de saída em probabilidades de 0 a 1 (sem precisar do PyTorch)."""
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))


def infer_probs(
    model_or_session: Any,
    img_bgr: np.ndarray,
    device: Any = None,
    res: int = 1024,
) -> np.ndarray:
    """
    Roda o modelo (PyTorch ou ONNX) e devolve probabilidades `(3, H, W)` no tamanho original da foto.
    Canais: 0 = copa, 1 = fiação, 2 = poste.
    """
    orig_h, orig_w = img_bgr.shape[:2]
    inp_np = _preprocess_1024(img_bgr, res=res)

    # Caminho 1: modelo leve quantizado rodando via ONNX Runtime
    if hasattr(model_or_session, "get_inputs"):
        input_name = model_or_session.get_inputs()[0].name
        # O SegFormer sempre devolve mapas em 1/4 da resolução de entrada (ex.: 256x256 para 1024x1024)
        logits = model_or_session.run(None, {input_name: inp_np})[0]
        probs_low = _sigmoid_numpy(logits)[0]
        return np.stack(
            [
                cv2.resize(probs_low[c], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                for c in range(probs_low.shape[0])
            ],
            axis=0,
        )

    # Caminho 2: modelo PyTorch completo em GPU ou CPU
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
    """Devolve as máscaras binárias `(mask_tree, mask_wire)` usando cortes calibrados por classe."""
    probs = infer_probs(model_or_session, img_bgr, device=device, res=res)
    w_th = 0.35 if wire_thresh is None else wire_thresh
    mask_tree = (probs[0] >= tree_thresh).astype(np.uint8)
    mask_wire = (probs[1] >= w_th).astype(np.uint8)
    return mask_tree, mask_wire


# ── Motor geométrico V3 e consenso entre duas visadas da mesma árvore ────────
def analyze_view_v3_manifest(
    mask_tree: np.ndarray,
    mask_wire: np.ndarray,
    file_name: str,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """
    Mede quantos pixels de fiação tocam a copa (r1 = 6 px) ou ficam na faixa de vento (r2 = 24 px)
    e retorna o resumo `v3_analysis` no formato esperado pelo `manifest.json`.
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
    Cruza os dois ângulos da mesma árvore para evitar falsos alarmes causados pela perspectiva:
    se apenas uma foto mostrar contato e a foto lateral mostrar árvore limpa, classifica como
    `DIVERGENCIA_PARALAXE` em vez de confirmar `CRITICO_CONFIRMADO`.
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
        # Se a segunda foto pelo menos confirmar proximidade, mantém como crítico confirmado
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
