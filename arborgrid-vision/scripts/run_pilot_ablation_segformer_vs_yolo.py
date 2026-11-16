"""
ArborGrid - Benchmark Piloto Controlado (Ablation de Qualidade de Rótulos)

Pergunta: as 24 imagens revisadas à mão (< tree_605898) melhoram o modelo em relação ao retoque do Lorenzo
e ao pré-label automático da GPU?

Desenho experimental (a única variável é a origem dos rótulos das imagens REVISADAS de treino):
  - Treino : 24 imagens revisadas (12 árvores) + 39 imagens ainda NÃO revisadas (>= tree_605898)
             * Modelo A: rótulos = pré-label GPU (pacote_colega_1)
             * Modelo B: rótulos = Lorenzo (pacote_colega_1_labelled)
             * Modelo C: rótulos = SUA revisão nas 24 revisadas + Lorenzo nas 39 não revisadas
  - Validação: 8 imagens (4 árvores isoladas por ID, sem vazamento) com SEU gabarito (pacote_colega_1_revised)
  - Arquitetura: SegFormer-B2 multi-label (3 canais sigmoid independentes: copa/fiacao/poste, BCE + Dice)
  - Sem seleção de checkpoint pela validação (usa a última época) => sem vazamento de seleção
  - Média de várias sementes (--seeds) para estimar o ruído de um conjunto de validação minúsculo

Extras:
  - IoU direto entre as anotações (pré-label / Lorenzo vs. você) nas imagens revisadas
  - Comparativo opcional com YOLOv8s-Seg estabilizado (freeze=10)
  - Scanner de anomalias Out-of-Fold (2-fold por árvore) sobre TODAS as imagens do pacote, priorizando o que revisar
"""

import os
import json
import random
import shutil
import argparse
import time
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerForSemanticSegmentation

CV_ROOT = Path(__file__).resolve().parent.parent
WS_DIR = CV_ROOT / "dataset" / "workspace_xanylabeling"
MODELS_DIR = CV_ROOT / "models"
ABLATION_DIR = CV_ROOT / "runs" / "pilot_ablation"

PACK_BASE = WS_DIR / "pacote_colega_1"
PACK_LORENZO = WS_DIR / "pacote_colega_1_labelled"
PACK_REVISED = WS_DIR / "pacote_colega_1_revised"

CUTOFF_NAME = "tree_605898"  # revisadas = nomes < cutoff
CLASSES = ["copa", "fiacao", "poste"]
CLASS_ID = {"copa": 0, "fiacao": 1, "poste": 2}
COLORS_BGR = {0: (60, 195, 45), 1: (35, 45, 245), 2: (235, 165, 20)}

VAL_TREE_IDS = {"597616", "597652", "597668", "597842"}
SEGFORMER_PRETRAINED = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

AMP = torch.cuda.is_available()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------
# Rótulos -> máscaras
# ----------------------------------------------------------------------------

def ensure_min_wire_thickness(pts, w, h, min_thick_px: float = 4.0):
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


def json_to_class_masks(json_path: Path, target_h: int = 640, target_w: int = 640) -> dict:
    masks = {c: np.zeros((target_h, target_w), dtype=np.uint8) for c in range(3)}
    d = json.loads(json_path.read_text(encoding="utf-8"))
    orig_w = float(d.get("imageWidth") or 640)
    orig_h = float(d.get("imageHeight") or 640)
    sx = target_w / max(orig_w, 1.0)
    sy = target_h / max(orig_h, 1.0)

    for sh in d.get("shapes") or []:
        lbl = str(sh.get("label") or "").strip().lower()
        if lbl not in CLASS_ID:
            continue
        cid = CLASS_ID[lbl]
        raw_pts = sh.get("points") or []
        if len(raw_pts) >= 3:
            pts = [[p[0] * sx, p[1] * sy] for p in raw_pts]
            if cid == 1:
                pts = ensure_min_wire_thickness(pts, target_w, target_h, min_thick_px=4.5)
            cv2.fillPoly(masks[cid], [np.round(np.array(pts)).astype(np.int32)], 1)
    return masks


# ----------------------------------------------------------------------------
# Dataset / Loss / Treino
# ----------------------------------------------------------------------------

class ArborGridMultiLabelDataset(Dataset):
    """items = lista de (stem, pasta_dos_rotulos). As imagens sempre vêm de PACK_REVISED (jpgs idênticos)."""

    def __init__(self, items, augment: bool = False, imgsz: int = 800):
        self.augment = augment
        self.imgsz = imgsz
        self.items = []
        for stem, lbl_dir in items:
            img_bgr = cv2.imread(str(PACK_REVISED / f"{stem}.jpg"))
            if img_bgr.shape[:2] != (imgsz, imgsz):
                img_bgr = cv2.resize(img_bgr, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            m = json_to_class_masks(lbl_dir / f"{stem}.json", imgsz, imgsz)
            self.items.append((stem, img_rgb, np.stack([m[0], m[1], m[2]], axis=-1)))  # (H, W, 3) uint8

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        stem, img, msk = self.items[idx]
        img, msk = img.copy(), msk.copy()
        if self.augment:
            if random.random() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1, :])
                msk = np.ascontiguousarray(msk[:, ::-1, :])
            if random.random() < 0.6:
                a, b = random.uniform(0.85, 1.18), random.uniform(-14.0, 14.0)
                img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
            if random.random() < 0.4:
                h, w = img.shape[:2]
                s = random.uniform(0.84, 1.0)
                nh, nw = int(h * s), int(w * s)
                y0, x0 = random.randint(0, h - nh), random.randint(0, w - nw)
                img = cv2.resize(img[y0:y0 + nh, x0:x0 + nw], (w, h), interpolation=cv2.INTER_LINEAR)
                msk = cv2.resize(msk[y0:y0 + nh, x0:x0 + nw], (w, h), interpolation=cv2.INTER_NEAREST)

        img_f = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(np.transpose(img_f, (2, 0, 1)))
        y = torch.from_numpy(np.transpose(msk, (2, 0, 1)).astype(np.float32))
        return x, y


class MultiLabelSegLoss(nn.Module):
    """BCE ponderada por raridade + Soft Dice por canal (peso maior na fiação)."""

    def __init__(self, pos_weights=(1.2, 8.0, 5.0)):
        super().__init__()
        self.register_buffer("pw", torch.tensor(pos_weights, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pw)
        p = torch.sigmoid(logits)
        inter = (p * targets).sum(dim=(0, 2, 3))
        denom = p.sum(dim=(0, 2, 3)) + targets.sum(dim=(0, 2, 3)) + 1.0
        dice_c = 1.0 - (2.0 * inter + 1.0) / denom
        dice = 0.25 * dice_c[0] + 0.50 * dice_c[1] + 0.25 * dice_c[2]
        return 0.45 * bce + 0.55 * dice


# ----------------------------------------------------------------------------
# Dashboard de Progresso em Tempo Real (HTML + JSON)
# ----------------------------------------------------------------------------

def write_html_dashboard(status: dict):
    out_html = ABLATION_DIR / "train_progress.html"
    out_json = ABLATION_DIR / "train_status.json"

    out_json.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

    pct = status.get("overall_pct", 0.0)
    stage = status.get("stage", "Inicializando...")
    current_model = status.get("current_model", "")
    ep = status.get("epoch", 0)
    total_ep = status.get("total_epochs", 0)
    loss = status.get("loss", 0.0)
    elapsed_m = status.get("elapsed_min", 0.0)
    eta_m = status.get("eta_min", 0.0)
    done = status.get("completed", False)
    metrics = status.get("metrics", {})

    meta_refresh = '<meta http-equiv="refresh" content="3">' if not done else ''
    badge_status = '<span class="badge bg-green">CONCLUÍDO</span>' if done else '<span class="badge bg-blue pulse">RODANDO</span>'

    table_rows = []
    labels = {"A": "Modelo A (Pré-label GPU)", "B": "Modelo B (Lorenzo)", "C": "Modelo C (Sua Revisão 100%)"}
    for k in ("A", "B", "C"):
        if k in metrics:
            m = metrics[k]
            row = f"""
            <tr>
              <td><strong>{labels[k]}</strong></td>
              <td class="text-accent">{m['fiacao']['iou']:.1f}%</td>
              <td>{m['fiacao']['recall']:.1f}%</td>
              <td>{m['fiacao']['precision']:.1f}%</td>
              <td>{m['copa']['iou']:.1f}%</td>
              <td>{m['poste']['iou']:.1f}%</td>
              <td class="text-bold">{m['mean_iou']:.1f}%</td>
            </tr>
            """
            table_rows.append(row)
        else:
            table_rows.append(f"""
            <tr class="dimmed">
              <td>{labels[k]}</td>
              <td colspan="6"><em>Aguardando execução...</em></td>
            </tr>
            """)

    html_content = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  {meta_refresh}
  <title>ArborGrid - Monitor de Treinamento</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: #0f172a;
      color: #e2e8f0;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      padding: 24px;
      line-height: 1.5;
    }}
    .container {{ max-width: 960px; margin: 0 auto; }}
    .header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 24px;
      border-bottom: 1px solid #1e293b;
      padding-bottom: 16px;
    }}
    .title h1 {{ font-size: 1.6rem; color: #f8fafc; font-weight: 700; }}
    .title p {{ font-size: 0.9rem; color: #94a3b8; margin-top: 4px; }}
    .badge {{
      display: inline-block;
      padding: 4px 12px;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 700;
      letter-spacing: 0.05em;
    }}
    .bg-blue {{ background: #0284c7; color: #f0f9ff; }}
    .bg-green {{ background: #16a34a; color: #f0fdf4; }}
    .pulse {{ animation: pulse 2s infinite; }}
    @keyframes pulse {{
      0%, 100% {{ opacity: 1; }}
      50% {{ opacity: 0.6; }}
    }}
    .card {{
      background: #1e293b;
      border-radius: 12px;
      padding: 20px;
      margin-bottom: 20px;
      border: 1px solid #334155;
    }}
    .progress-bar-bg {{
      background: #334155;
      border-radius: 9999px;
      height: 18px;
      overflow: hidden;
      margin: 12px 0 6px;
    }}
    .progress-bar-fill {{
      background: linear-gradient(90deg, #06b6d4, #10b981);
      height: 100%;
      border-radius: 9999px;
      transition: width 0.4s ease;
    }}
    .progress-text {{
      display: flex;
      justify-content: space-between;
      font-size: 0.85rem;
      color: #94a3b8;
    }}
    .stats-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-top: 16px;
    }}
    .stat-box {{
      background: #0f172a;
      padding: 12px 16px;
      border-radius: 8px;
      border: 1px solid #1e293b;
    }}
    .stat-label {{ font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }}
    .stat-val {{ font-size: 1.3rem; font-weight: 700; color: #f8fafc; margin-top: 4px; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 12px;
      font-size: 0.9rem;
    }}
    th, td {{
      padding: 10px 14px;
      text-align: left;
      border-bottom: 1px solid #334155;
    }}
    th {{ color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; }}
    .text-accent {{ color: #38bdf8; font-weight: 700; }}
    .text-bold {{ font-weight: 700; color: #f8fafc; }}
    .dimmed {{ opacity: 0.45; }}
    .footer {{
      text-align: center;
      margin-top: 24px;
      font-size: 0.8rem;
      color: #64748b;
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div class="title">
        <h1>ArborGrid — Monitor de Treinamento</h1>
        <p>RTX 3060 Ti &bull; SegFormer-B2 &bull; 68 Imagens Curadas</p>
      </div>
      <div>{badge_status}</div>
    </div>

    <div class="card">
      <h2 style="font-size: 1.1rem; color: #f8fafc;">{stage}</h2>
      <div class="progress-bar-bg">
        <div class="progress-bar-fill" style="width: {pct:.1f}%;"></div>
      </div>
      <div class="progress-text">
        <span>Progresso Total: <strong>{pct:.1f}%</strong></span>
        <span>Época: <strong>{ep}/{total_ep}</strong> (Modelo: {current_model})</span>
      </div>

      <div class="stats-grid">
        <div class="stat-box">
          <div class="stat-label">Loss Médio</div>
          <div class="stat-val text-accent">{loss:.4f}</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Tempo Decorrido</div>
          <div class="stat-val">{elapsed_m:.1f} min</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Tempo Restante (ETA)</div>
          <div class="stat-val" style="color: #34d399;">{eta_m:.1f} min</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Dispositivo</div>
          <div class="stat-val" style="font-size: 1.05rem;">cuda:0 (AMP)</div>
        </div>
      </div>
    </div>

    <div class="card">
      <h3 style="font-size: 1.05rem; margin-bottom: 8px;">Tabela Comparativa na Validação (Gabarito Ouro)</h3>
      <table>
        <thead>
          <tr>
            <th>Modelo</th>
            <th>IoU Fiação</th>
            <th>Recall Fiação</th>
            <th>Prec. Fiação</th>
            <th>IoU Copa</th>
            <th>IoU Poste</th>
            <th>mIoU Geral</th>
          </tr>
        </thead>
        <tbody>
          {''.join(table_rows)}
        </tbody>
      </table>
    </div>

    <div class="footer">
      Esta página auto-atualiza a cada 3 segundos enquanto o treino estiver ativo.<br>
      ArborGrid &bull; FIAP Global Solution 2026.2
    </div>
  </div>
</body>
</html>
"""
    out_html.write_text(html_content, encoding="utf-8")


def train_segformer(items, device: str, epochs: int, seed: int, batch_size: int = 2, imgsz: int = 800, ep_callback=None):
    """Treina na última época (sem escolher checkpoint pela validação)."""
    seed_everything(seed)
    loader = DataLoader(ArborGridMultiLabelDataset(items, augment=True, imgsz=imgsz), batch_size=batch_size, shuffle=True)
    model = SegformerForSemanticSegmentation.from_pretrained(
        SEGFORMER_PRETRAINED, num_labels=3, ignore_mismatched_sizes=True
    ).to(device)
    criterion = MultiLabelSegLoss().to(device)
    opt = torch.optim.AdamW(
        [
            {"params": model.segformer.parameters(), "lr": 1.2e-4},
            {"params": model.decode_head.parameters(), "lr": 6.0e-4},
        ],
        weight_decay=1e-2,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP)

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        run = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=AMP):
                out = model(pixel_values=x).logits
                loss = criterion(F.interpolate(out, size=y.shape[-2:], mode="bilinear", align_corners=False), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run += float(loss.item())
        sched.step()
        elapsed = time.time() - t0
        avg_loss = run / max(len(loader), 1)

        if ep_callback:
            ep_callback(ep, epochs, avg_loss, elapsed)

        if epochs <= 10 or ep % 5 == 0 or ep == epochs or ep == 1:
            print(f"      ép {ep:3d}/{epochs} ({elapsed:4.1f}s) | loss médio = {avg_loss:.4f}", flush=True)
    model.eval()
    return model


@torch.no_grad()
def predict(model, img_path: Path, device: str, imgsz: int = 800, thresh=(0.50, 0.40, 0.50)):
    orig_bgr = cv2.imread(str(img_path))
    h, w = orig_bgr.shape[:2]
    if (h, w) != (imgsz, imgsz):
        img_in = cv2.resize(orig_bgr, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    else:
        img_in = orig_bgr
    img_rgb = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
    img_f = (img_rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    x = torch.from_numpy(np.transpose(img_f, (2, 0, 1))).unsqueeze(0).to(device)
    with torch.amp.autocast("cuda", enabled=AMP):
        logits = model(pixel_values=x).logits
    probs = torch.sigmoid(F.interpolate(logits.float(), size=(h, w), mode="bilinear", align_corners=False))[0].cpu().numpy()
    return {c: (probs[c] >= thresh[c]).astype(np.uint8) for c in range(3)}, {c: probs[c] for c in range(3)}


# ----------------------------------------------------------------------------
# Métricas
# ----------------------------------------------------------------------------

def compute_iou(pred_list, gt_list) -> dict:
    out = {}
    for cid, name in enumerate(CLASSES):
        tp = fp = fn = 0
        for pm, gm in zip(pred_list, gt_list):
            p, g = pm[cid] > 0, gm[cid] > 0
            tp += int(np.count_nonzero(p & g))
            fp += int(np.count_nonzero(p & ~g))
            fn += int(np.count_nonzero(~p & g))
        out[name] = {
            "iou": round(100 * tp / max(tp + fp + fn, 1), 2),
            "precision": round(100 * tp / max(tp + fp, 1), 2),
            "recall": round(100 * tp / max(tp + fn, 1), 2),
        }
    out["mean_iou"] = round(float(np.mean([out[c]["iou"] for c in CLASSES])), 2)
    return out


def average_metrics(ms: list) -> dict:
    out = {c: {k: round(float(np.mean([m[c][k] for m in ms])), 2) for k in ms[0][c]} for c in CLASSES}
    out["mean_iou"] = round(float(np.mean([m["mean_iou"] for m in ms])), 2)
    out["mean_iou_per_seed"] = [m["mean_iou"] for m in ms]
    return out


def fmt(m: dict) -> str:
    return (
        f"copa {m['copa']['iou']:5.1f} | fiação {m['fiacao']['iou']:5.1f} "
        f"(P {m['fiacao']['precision']:5.1f} / R {m['fiacao']['recall']:5.1f}) | "
        f"poste {m['poste']['iou']:5.1f} | mIoU {m['mean_iou']:5.1f}"
    )


def render_panel(img_bgr, masks, title):
    canvas, overlay = img_bgr.copy(), img_bgr.copy()
    for cid in (0, 2, 1):
        if np.any(masks[cid]):
            overlay[masks[cid] > 0] = COLORS_BGR[cid]
    cv2.addWeighted(overlay, 0.44, canvas, 0.56, 0, canvas)
    for cid in (0, 2, 1):
        cnts, _ = cv2.findContours(masks[cid], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cv2.drawContours(canvas, cnts, -1, COLORS_BGR[cid], 2 if cid != 0 else 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 28), (15, 23, 42), -1)
    cv2.putText(canvas, title, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (248, 250, 252), 1, cv2.LINE_AA)
    return canvas


# ----------------------------------------------------------------------------
# YOLO opcional (comparativo arquitetural)
# ----------------------------------------------------------------------------

def run_yolo_comparison(items_c, val_stems):
    from ultralytics import YOLO

    base_pt = MODELS_DIR / "yolov8s-seg.pt"
    if not base_pt.exists():
        YOLO("yolov8s-seg.pt")
        dl = CV_ROOT / "yolov8s-seg.pt"
        if dl.exists():
            shutil.move(str(dl), str(base_pt))

    ds = ABLATION_DIR / "datasets" / "ds_C_yolo"
    if ds.exists():
        shutil.rmtree(ds)
    for sp in ("train", "val"):
        (ds / "images" / sp).mkdir(parents=True, exist_ok=True)
        (ds / "labels" / sp).mkdir(parents=True, exist_ok=True)

    def write(stem, lbl_dir, sp):
        shutil.copy2(PACK_REVISED / f"{stem}.jpg", ds / "images" / sp / f"{stem}.jpg")
        d = json.loads((lbl_dir / f"{stem}.json").read_text(encoding="utf-8"))
        w, h = int(d.get("imageWidth") or 640), int(d.get("imageHeight") or 640)
        lines = []
        for sh in d.get("shapes") or []:
            lbl = str(sh.get("label") or "").strip().lower()
            pts = sh.get("points") or []
            if lbl in CLASS_ID and len(pts) >= 3:
                cid = CLASS_ID[lbl]
                if cid == 1:
                    pts = ensure_min_wire_thickness(pts, w, h)
                lines.append(f"{cid} " + " ".join(f"{min(1, max(0, x / w)):.6f} {min(1, max(0, y / h)):.6f}" for x, y in pts))
        (ds / "labels" / sp / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")

    for stem, lbl_dir in items_c:
        write(stem, lbl_dir, "train")
    for stem in val_stems:
        write(stem, PACK_REVISED, "val")

    (ds / "data.yaml").write_text(
        f"path: {ds.as_posix()}\ntrain: images/train\nval: images/val\n\nnc: 3\nnames:\n  0: copa\n  1: fiacao\n  2: poste\n",
        encoding="utf-8",
    )
    ym = YOLO(str(base_pt))
    ym.train(
        data=str(ds / "data.yaml"), epochs=100, imgsz=640, batch=8, freeze=10, lr0=0.0006, optimizer="AdamW",
        mask_ratio=2, overlap_mask=False, patience=0, workers=0, seed=42,
        project=str(ABLATION_DIR / "train_runs"), name="yolo_C", exist_ok=True, verbose=False,
    )
    best = YOLO(str(ABLATION_DIR / "train_runs" / "yolo_C" / "weights" / "best.pt"))
    preds = []
    for s in val_stems:
        res = best.predict(str(PACK_REVISED / f"{s}.jpg"), imgsz=640, conf=0.20, retina_masks=True, verbose=False)[0]
        bm = {c: np.zeros((640, 640), dtype=np.uint8) for c in range(3)}
        if res.masks is not None and res.boxes is not None:
            for m2d, cid in zip(res.masks.data.cpu().numpy(), res.boxes.cls.cpu().numpy().astype(int)):
                if m2d.shape != (640, 640):
                    m2d = cv2.resize(m2d, (640, 640))
                bm[cid][m2d > 0.5] = 1
        preds.append(bm)
    return preds


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--imgsz", type=int, default=800, help="Resolução de entrada para o SegFormer (default: 800)")
    ap.add_argument("--skip-yolo", action="store_true", default=True)
    ap.add_argument("--skip-oof", action="store_true", default=True)
    ap.add_argument("--run-oof", action="store_true", help="Executa scanner OOF de anomalias")
    ap.add_argument("--sanity", action="store_true", help="Teste rápido de sanidade (3 épocas, 1 seed, sem YOLO/OOF)")
    args = ap.parse_args()

    if args.sanity:
        args.epochs = 3
        args.seeds = [42]
        args.skip_yolo = True
        args.skip_oof = True
    elif args.run_oof:
        args.skip_oof = False

    os.chdir(CV_ROOT)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    all_stems = [p.stem for p in sorted(PACK_REVISED.glob("*.json"))]
    val_stems = [s for s in all_stems if s.split("_")[1] in VAL_TREE_IDS]
    train_stems = [s for s in all_stems if s.split("_")[1] not in VAL_TREE_IDS]
    reviewed = all_stems

    print("=" * 80, flush=True)
    print(f"BENCHMARK COMPLETO (100% REVISADO) | device={device} | seeds={args.seeds} | épocas={args.epochs} | batch={args.batch_size} | imgsz={args.imgsz}x{args.imgsz}", flush=True)
    print(f"  Total imagens no pacote : {len(all_stems)} (todas 100% revisadas pelo usuário!)", flush=True)
    print(f"  Divisão sem vazamento   : Treino = {len(train_stems)} fotos | Validação = {len(val_stems)} fotos ({len(VAL_TREE_IDS)} árvores isoladas)", flush=True)
    print("=" * 80, flush=True)

    def items_for(variant):
        rev_dir = {"A": PACK_BASE, "B": PACK_LORENZO, "C": PACK_REVISED}[variant]
        return [(s, rev_dir) for s in train_stems]

    # 1) IoU direto entre anotações nas imagens revisadas (todas as 68)
    gt_all = [json_to_class_masks(PACK_REVISED / f"{s}.json") for s in all_stems]
    direct_base = compute_iou([json_to_class_masks(PACK_BASE / f"{s}.json") for s in all_stems], gt_all)
    direct_lor = compute_iou([json_to_class_masks(PACK_LORENZO / f"{s}.json") for s in all_stems], gt_all)

    # 2) Treino A/B/C x sementes
    gt_val = [json_to_class_masks(PACK_REVISED / f"{s}.json") for s in val_stems]
    per_variant = {"A": [], "B": [], "C": []}
    preds_seed0 = {}
    latest_metrics = {}

    total_models = len(args.seeds) * 3
    total_epochs_all = total_models * args.epochs
    start_global = time.time()
    steps_done = 0

    write_html_dashboard({
        "stage": "Iniciando Treinamento...",
        "current_model": "-",
        "epoch": 0,
        "total_epochs": args.epochs,
        "loss": 0.0,
        "overall_pct": 0.0,
        "elapsed_min": 0.0,
        "eta_min": (total_epochs_all * 13.5) / 60.0,
        "completed": False,
        "metrics": latest_metrics,
    })

    for si, seed in enumerate(args.seeds):
        for v_idx, (v, name) in enumerate((("A", "Base (pré-label GPU)"), ("B", "Lorenzo"), ("C", "Sua revisão"))):
            model_num = si * 3 + v_idx + 1

            def ep_cb(ep, tot_ep, loss, ep_time):
                nonlocal steps_done
                steps_done = (model_num - 1) * args.epochs + ep
                pct = min(100.0, (steps_done / total_epochs_all) * 100.0)
                elapsed = time.time() - start_global
                pace = elapsed / max(steps_done, 1)
                eta = (total_epochs_all - steps_done) * pace
                write_html_dashboard({
                    "stage": f"Treinando {name} (Modelo {v}) @ {args.imgsz}x{args.imgsz}",
                    "current_model": f"{v} ({name})",
                    "epoch": ep,
                    "total_epochs": tot_ep,
                    "loss": loss,
                    "overall_pct": pct,
                    "elapsed_min": elapsed / 60.0,
                    "eta_min": eta / 60.0,
                    "completed": False,
                    "metrics": latest_metrics,
                })

            print(f"[{model_num}/{total_models}] Treinando Modelo {v} - {name} ({len(items_for(v))} imagens @ {args.imgsz}x{args.imgsz})...", flush=True)
            model = train_segformer(items_for(v), device, args.epochs, seed, batch_size=args.batch_size, imgsz=args.imgsz, ep_callback=ep_cb)
            preds = [predict(model, PACK_REVISED / f"{s}.jpg", device, imgsz=args.imgsz)[0] for s in val_stems]
            m = compute_iou(preds, gt_val)
            per_variant[v].append(m)
            latest_metrics[v] = m
            print(f"      -> {fmt(m)}", flush=True)
            if si == 0:
                preds_seed0[v] = preds
                if v == "C":
                    torch.save(model.state_dict(), MODELS_DIR / "arborgrid_segformer_b2_pilot.pt")
            del model
            torch.cuda.empty_cache()

            # Salva progresso incremental a cada modelo finalizado
            partial = {
                "last_completed": f"seed_{seed}_model_{v}",
                "per_variant": per_variant,
            }
            (ABLATION_DIR / "ablation_metrics_partial.json").write_text(
                json.dumps(partial, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    avg = {v: average_metrics(per_variant[v]) for v in per_variant}

    # 3) YOLO opcional
    yolo_metrics = None
    if not args.skip_yolo:
        print("[YOLO] Comparativo arquitetural (YOLOv8s-Seg, freeze=10) no Dataset C...", flush=True)
        try:
            yolo_metrics = compute_iou(run_yolo_comparison(items_for("C"), val_stems), gt_val)
            print(f"      -> {fmt(yolo_metrics)}", flush=True)
        except Exception as e:
            print(f"      YOLO falhou e foi ignorado: {e}", flush=True)

    # 4) Visual comparativo (seed 0)
    vis_dir = ABLATION_DIR / "val_comparisons"
    if vis_dir.exists():
        shutil.rmtree(vis_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, stem in enumerate(val_stems):
        img = cv2.imread(str(PACK_REVISED / f"{stem}.jpg"))
        tiles = [
            render_panel(img, gt_val[i], f"SEU GABARITO: {stem[:28]}"),
            render_panel(img, preds_seed0["A"][i], "Modelo A (pre-label)"),
            render_panel(img, preds_seed0["B"][i], "Modelo B (Lorenzo)"),
            render_panel(img, preds_seed0["C"][i], "Modelo C (sua revisao)"),
        ]
        strip = np.hstack([cv2.resize(t, (320, 320)) for t in tiles])
        cv2.imwrite(str(vis_dir / f"val_{stem}.jpg"), strip, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        rows.append(strip)
    cv2.imwrite(str(ABLATION_DIR / "mosaico_comparativo_modelos_val.jpg"), np.vstack(rows), [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    # 5) Scanner de anomalias Out-of-Fold em TODAS as imagens (2-fold por árvore)
    anomalies = []
    if not args.skip_oof:
        print("[OOF] Scanner de anomalias (2-fold por árvore) sobre todas as imagens...", flush=True)
        tids = sorted({s.split("_")[1] for s in all_stems})
        fold_a = set(tids[::2])
        stems_a = [s for s in all_stems if s.split("_")[1] in fold_a]
        stems_b = [s for s in all_stems if s.split("_")[1] not in fold_a]
        m_a = train_segformer([(s, PACK_REVISED) for s in stems_a], device, args.epochs, args.seeds[0], batch_size=args.batch_size, imgsz=args.imgsz)  # treina em A, prevê B
        torch.cuda.empty_cache()
        m_b = train_segformer([(s, PACK_REVISED) for s in stems_b], device, args.epochs, args.seeds[0], batch_size=args.batch_size, imgsz=args.imgsz)  # treina em B, prevê A
        torch.cuda.empty_cache()

        anom_dir = ABLATION_DIR / "anomaly_candidates"
        if anom_dir.exists():
            shutil.rmtree(anom_dir)
        anom_dir.mkdir(parents=True, exist_ok=True)
        k_gt = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23))
        k_pr = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

        for stem in all_stems:
            predictor = m_a if stem in stems_b else m_b
            pm, cm = predict(predictor, PACK_REVISED / f"{stem}.jpg", device, imgsz=args.imgsz)
            gm = json_to_class_masks(PACK_REVISED / f"{stem}.json")
            gt_near = cv2.dilate(gm[1], k_gt)
            unl = (gt_near == 0) & (cm[1] >= 0.58)
            unl_px = int(np.count_nonzero(unl))
            gt_px = int(np.count_nonzero(gm[1]))
            cov = int(np.count_nonzero((gm[1] > 0) & (cv2.dilate(pm[1], k_pr) > 0)))
            rec = cov / max(gt_px, 1)

            reasons, sev = [], 0.0
            if unl_px >= 650:
                reasons.append(f"FIO PROVÁVEL NÃO ROTULADO ({unl_px} px com conf máx {float(np.max(cm[1][unl])):.2f} longe dos seus polígonos)")
                sev += unl_px / 350.0
            if gt_px >= 1500 and rec < 0.22:
                reasons.append(f"FIO ANOTADO DIFÍCIL/ATÍPICO (a IA out-of-fold só recuperou {rec * 100:.1f}% dos fios anotados)")
                sev += (1.0 - rec) * 3.5
            if reasons:
                img = cv2.imread(str(PACK_REVISED / f"{stem}.jpg"))
                out_a = anom_dir / f"anom_{stem}.jpg"
                cv2.imwrite(
                    str(out_a),
                    np.hstack([img, render_panel(img, gm, f"ROTULO ATUAL: {stem[:26]}"), render_panel(img, pm, "IA OUT-OF-FOLD")]),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 90],
                )
                anomalies.append({
                    "stem": stem, "reviewed_by_you": True, "severity": round(sev, 2),
                    "gt_wire_px": gt_px, "unlabeled_pred_wire_px": unl_px,
                    "oof_wire_recall_pct": round(rec * 100, 1), "reasons": reasons,
                    "preview": out_a.relative_to(CV_ROOT).as_posix(),
                })
        anomalies.sort(key=lambda a: a["severity"], reverse=True)

        if anomalies:
            top_anoms = anomalies[:6]
            anom_tiles = []
            for an in top_anoms:
                p_tile = anom_dir / f"anom_{an['stem']}.jpg"
                if p_tile.exists():
                    t_img = cv2.imread(str(p_tile))
                    h_t, w_t = t_img.shape[:2]
                    scale = 960.0 / w_t
                    t_res = cv2.resize(t_img, (960, int(h_t * scale)))
                    anom_tiles.append(t_res)
            if anom_tiles:
                cv2.imwrite(
                    str(ABLATION_DIR / "mosaico_top_anomalias.jpg"),
                    np.vstack(anom_tiles),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 90],
                )

    summary = {
        "config": {"seeds": args.seeds, "epochs": args.epochs,
                   "n_train": len(train_stems), "n_val": len(val_stems),
                   "val_tree_ids": sorted(VAL_TREE_IDS)},
        "direct_label_iou_reviewed_imgs": {"base_vs_revised": direct_base, "lorenzo_vs_revised": direct_lor},
        "segformer_val_iou_mean_over_seeds": avg,
        "segformer_val_iou_per_seed": per_variant,
        "yolov8s_seg_val_iou": yolo_metrics,
        "anomalies_oof": anomalies,
    }
    (ABLATION_DIR / "ablation_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    write_html_dashboard({
        "stage": "Treinamento Finalizado com Sucesso!",
        "current_model": "Concluído",
        "epoch": args.epochs,
        "total_epochs": args.epochs,
        "loss": 0.0,
        "overall_pct": 100.0,
        "elapsed_min": (time.time() - start_global) / 60.0,
        "eta_min": 0.0,
        "completed": True,
        "metrics": latest_metrics,
    })

    print("\n" + "=" * 80, flush=True)
    print("RESULTADOS", flush=True)
    print("=" * 80, flush=True)
    print(f"[A] IoU direto entre rótulos nas {len(all_stems)} imagens revisadas (vs. seu gabarito):", flush=True)
    print(f"    Pré-label GPU : {fmt(direct_base)}", flush=True)
    print(f"    Lorenzo       : {fmt(direct_lor)}", flush=True)
    print(f"\n[B] SegFormer-B2 na validação ({len(val_stems)} imgs), média de {len(args.seeds)} sementes:", flush=True)
    for v, nm in (("A", "Treinado no pré-label"), ("B", "Treinado no Lorenzo "), ("C", "Treinado na sua rev.")):
        print(f"    {nm}: {fmt(avg[v])}   (mIoU por semente: {avg[v]['mean_iou_per_seed']})", flush=True)
    if yolo_metrics:
        print(f"\n[C] YOLOv8s-Seg (referência arquitetural): {fmt(yolo_metrics)}", flush=True)
    if anomalies:
        print(f"\n[D] Anomalias out-of-fold ({len(anomalies)} de {len(all_stems)} fotos) - top 12:", flush=True)
        for a in anomalies[:12]:
            print(f"    * {a['stem']} score={a['severity']}: {' | '.join(a['reasons'])}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()

