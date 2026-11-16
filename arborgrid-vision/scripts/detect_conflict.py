"""
ArborGrid - Detecção Espacial de Conflito Rede Elétrica vs. Vegetação Urbana
Módulo: arborgrid-vision / scripts / detect_conflict.py

Este script analisa imagens de vias urbanas contendo postes elétricos e árvores,
calculando a intersecção geométrica entre a zona de fiação (parte superior dos postes)
e a vegetação/copas para gerar alertas automatizados de risco de queda/rompimento.
"""

import os
import glob
import json
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def calculate_box_intersection(boxA, boxB):
    """
    Calcula se há sobreposição entre duas bounding boxes e retorna a área e coordenadas da interseção.
    Formato das boxes: [x1, y1, x2, y2] em coordenadas absolutas de pixel.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    if inter_area > 0:
        return True, [xA, yA, xB, yB], inter_area
    return False, None, 0

def get_pole_electrical_hazard_zone(pole_box, hazard_ratio=0.45, lateral_margin=0.6):
    """
    Define a zona crítica de fiação elétrica como os X% superiores do poste (por padrão 45%),
    onde se concentram os isoladores, cruzetas, transformadores e cabos de média/baixa tensão.
    Também expande ligeiramente nas laterais para cobrir os braços de iluminação e cruzetas.
    """
    x1, y1, x2, y2 = pole_box
    pole_height = y2 - y1
    hazard_y2 = y1 + (pole_height * hazard_ratio)
    
    pole_width = x2 - x1
    margin = pole_width * lateral_margin
    hz_x1 = max(0, x1 - margin)
    hz_x2 = x2 + margin

    return [hz_x1, y1, hz_x2, hazard_y2]

def analyze_image_encroachment(image_path, detections, output_path=None, hazard_ratio=0.45):
    """
    Analisa uma imagem dada uma lista de detecções:
    detections: lista de dicts com {'class_name': str, 'box': [x1, y1, x2, y2], 'conf': float}
    """
    img = Image.open(image_path).convert('RGBA')
    width, height = img.size

    poles = [d for d in detections if 'pole' in d['class_name'].lower() or 'poste' in d['class_name'].lower()]
    trees = [d for d in detections if 'tree' in d['class_name'].lower() or 'arvore' in d['class_name'].lower() or 'vegetacao' in d['class_name'].lower()]

    conflicts = []
    
    overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
    draw_ov = ImageDraw.Draw(overlay)
    draw = ImageDraw.Draw(img)

    # 1. Desenhar árvores (Verde)
    for tree in trees:
        tx1, ty1, tx2, ty2 = tree['box']
        draw.rectangle([tx1, ty1, tx2, ty2], outline=(46, 204, 113, 230), width=3)
        draw.text((tx1 + 4, ty1 + 4), f"Vegetacao ({tree.get('conf', 1.0):.2f})", fill=(46, 204, 113, 255))

    # 2. Desenhar postes e analisar zona elétrica
    for pole in poles:
        px1, py1, px2, py2 = pole['box']
        draw.rectangle([px1, py1, px2, py2], outline=(52, 152, 219, 230), width=3)
        draw.text((px1 + 4, py2 - 18), f"Poste Eletrico ({pole.get('conf', 1.0):.2f})", fill=(52, 152, 219, 255))

        # Calcular Zona de Fiação
        wire_zone = get_pole_electrical_hazard_zone([px1, py1, px2, py2], hazard_ratio=hazard_ratio)
        wx1, wy1, wx2, wy2 = wire_zone
        
        # Desenhar borda da zona de fiação
        draw.rectangle([wx1, wy1, wx2, wy2], outline=(241, 196, 15, 220), width=2)
        draw.text((wx1 + 4, wy1 + 4), "Zona de Cabos/Cruzeta", fill=(241, 196, 15, 255))

        # Verificar conflito com cada árvore detectada
        for tree in trees:
            has_inter, inter_box, area = calculate_box_intersection(wire_zone, tree['box'])
            if has_inter:
                ix1, iy1, ix2, iy2 = inter_box
                conflicts.append({
                    'pole_box': [px1, py1, px2, py2],
                    'tree_box': tree['box'],
                    'conflict_box': [ix1, iy1, ix2, iy2],
                    'intersection_area': area
                })
                # Destaque em vermelho da área de interseção física
                draw_ov.rectangle([ix1, iy1, ix2, iy2], fill=(231, 76, 60, 140), outline=(231, 76, 60, 255), width=3)
                draw.text((ix1 + 4, iy1 + 4), "! RISCO: ENGENHAMENTO NA REDE !", fill=(255, 255, 255, 255))

    # Definir Severidade
    if len(conflicts) > 0:
        risk_level = "CRITICO (ALERTA DEFESA CIVIL / ENEL)"
        header_color = (231, 76, 60, 225)
    elif len(poles) > 0 and len(trees) > 0:
        risk_level = "MODERADO (PROXIMIDADE MONITORADA)"
        header_color = (243, 156, 18, 225)
    else:
        risk_level = "SEGURO (SEM CONFLITO DETECTADO)"
        header_color = (46, 204, 113, 225)
    
    # Telemetria Superior
    draw_ov.rectangle([10, 10, 480, 70], fill=header_color, outline=(255, 255, 255, 255), width=2)
    draw.text((20, 16), f"ARBORGRID | {risk_level}", fill=(255, 255, 255, 255))
    draw.text((20, 40), f"Postes: {len(poles)} | Arvores: {len(trees)} | Conflitos Diretos: {len(conflicts)}", fill=(255, 255, 255, 255))

    final_img = Image.alpha_composite(img, overlay).convert('RGB')

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        final_img.save(output_path, quality=90)

    return {
        'image': str(image_path),
        'risk_level': risk_level,
        'poles_count': len(poles),
        'trees_count': len(trees),
        'conflicts_count': len(conflicts),
        'conflicts': conflicts
    }

def parse_yolo_txt_label(txt_path, img_width, img_height, class_names):
    detections = []
    if not os.path.exists(txt_path):
        return detections
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cid = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                x1 = int((cx - bw / 2) * img_width)
                y1 = int((cy - bh / 2) * img_height)
                x2 = int((cx + bw / 2) * img_width)
                y2 = int((cy + bh / 2) * img_height)
                
                cname = class_names[cid] if cid < len(class_names) else f"class_{cid}"
                detections.append({
                    'class_id': cid,
                    'class_name': cname,
                    'box': [x1, y1, x2, y2],
                    'conf': 1.0
                })
    return detections

def run_model_inference_on_folder(weights_path, image_dir, output_dir, conf=0.25):
    from ultralytics import YOLO
    print(f"Carregando modelo: {weights_path}")
    model = YOLO(weights_path)
    
    extensions = ('*.jpg', '*.jpeg', '*.png')
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
    
    print(f"Encontradas {len(image_paths)} imagens em {image_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    results_summary = []
    for img_p in image_paths:
        res = model.predict(source=img_p, conf=conf, verbose=False)[0]
        
        # Converter caixas detectadas
        detections = []
        for box in res.boxes:
            cid = int(box.cls[0].item())
            cname = res.names[cid]
            cconf = float(box.conf[0].item())
            xyxy = [int(v) for v in box.xyxy[0].tolist()]
            detections.append({
                'class_id': cid,
                'class_name': cname,
                'box': xyxy,
                'conf': cconf
            })
            
        base_name = os.path.basename(img_p)
        out_p = os.path.join(output_dir, f"analyzed_{base_name}")
        report = analyze_image_encroachment(img_p, detections, output_path=out_p)
        results_summary.append(report)
        print(f"[{report['risk_level']}] {base_name} -> {report['conflicts_count']} conflitos")

    summary_file = os.path.join(output_dir, "encroachment_report.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2)
    print(f"Relatorio salvo em: {summary_file}")
    return results_summary

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ArborGrid Encroachment Detector")
    parser.add_argument("--img", type=str, help="Caminho para imagem individual")
    parser.add_argument("--weights", type=str, help="Pesos .pt do YOLOv8")
    parser.add_argument("--dir", type=str, help="Pasta de imagens para inferir")
    parser.add_argument("--out_dir", type=str, default="preview/analyzed", help="Pasta de saida")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    if args.dir and args.weights:
        run_model_inference_on_folder(args.weights, args.dir, args.out_dir, args.conf)
    else:
        print("Use --dir com --weights para processar pastas de imagens.")
