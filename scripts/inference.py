"""
Script d'inférence unifié — YOLOv11 Segmentation & Mask R-CNN
Effectue la détection de fissures et l'analyse morphologique post-inférence.
"""

import os
import sys
import argparse
import cv2
import numpy as np
from pathlib import Path


# ─── Arguments ────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Inférence fissures (YOLOv11 ou Mask R-CNN)")
    parser.add_argument("--model", type=str, required=True,
                        help="Chemin vers le modèle : best.pt (YOLO) ou model_final.pth (Mask R-CNN)")
    parser.add_argument("--source", type=str, required=True,
                        help="Image, dossier d'images ou vidéo à analyser")
    parser.add_argument("--model-type", type=str, choices=["yolo", "maskrcnn"], default=None,
                        help="Type de modèle. Détecté automatiquement si absent.")
    parser.add_argument("--output-dir", type=str, default="outputs/inference",
                        help="Dossier de sortie pour les images annotées et rapports JSON")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Seuil de confiance (YOLO)")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="Seuil IoU pour NMS (YOLO)")
    parser.add_argument("--score-thresh", type=float, default=0.5,
                        help="Seuil de score (Mask R-CNN)")
    parser.add_argument("--num-classes", type=int, default=1,
                        help="Nombre de classes du modèle Mask R-CNN (doit correspondre au nombre "
                             "de catégories du JSON COCO utilisé à l'entraînement — voir le message "
                             "'Classes détectées dans le JSON COCO' affiché par train_maskrcnn.py)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Taille des images en entrée (YOLO)")
    parser.add_argument("--px-to-mm", type=float, default=0.5,
                        help="Facteur de calibration mm/pixel pour l'analyse")
    parser.add_argument("--device", type=str, default=None,
                        help="Appareil : 0, cpu ... (YOLO uniquement)")
    parser.add_argument("--no-analysis", action="store_true",
                        help="Désactiver l'analyse morphologique post-inférence")
    parser.add_argument("--save-masks", action="store_true",
                        help="Sauvegarder les masques bruts en plus des images annotées")
    return parser.parse_args()


# ─── Détection automatique du type de modèle ─────────────────────────────────

def detect_model_type(model_path: str) -> str:
    p = Path(model_path)
    if p.suffix == ".pt":
        return "yolo"
    elif p.suffix in (".pth", ".pkl"):
        return "maskrcnn"
    raise ValueError(f"Extension de modèle non reconnue : {p.suffix}. Utilisez --model-type.")


# ─── Collecte des sources ─────────────────────────────────────────────────────

def collect_sources(source: str):
    """Collecte les images depuis un fichier unique ou un dossier."""
    p = Path(source)
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    if p.is_file():
        if p.suffix.lower() in img_exts:
            return [p], "images"
        else:
            raise ValueError(
                f"Format source non supporté : {p.suffix}\n"
                f"Formats acceptés : {', '.join(sorted(img_exts))}"
            )
    elif p.is_dir():
        imgs = sorted([f for f in p.iterdir() if f.suffix.lower() in img_exts])
        if not imgs:
            raise FileNotFoundError(f"Aucune image trouvée dans : {p}")
        return imgs, "images"
    else:
        raise FileNotFoundError(f"Source introuvable : {source}")


# ─── Inférence YOLOv11 ────────────────────────────────────────────────────────

def run_yolo_inference(args, sources, output_dir: Path):
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("❌ ultralytics non installé : pip install ultralytics")

    from scripts.crack_analysis import analyze_frame, draw_analysis, save_analysis_report, print_analysis_summary

    print(f"\n▶ Chargement du modèle YOLOv11 : {args.model}")
    model = YOLO(args.model)

    results_all = []

    for img_path in sources:
        print(f"\n  Traitement : {img_path.name}")
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  ⚠ Impossible de lire : {img_path}")
            continue

        kwargs = {
            "source": str(img_path),
            "conf":   args.conf,
            "iou":    args.iou,
            "imgsz":  args.imgsz,
            "retina_masks": True,
            "verbose": False,
        }
        if args.device is not None:
            kwargs["device"] = args.device

        preds = model(**kwargs)

        for pred in preds:
            masks_list = []
            scores_list = []

            if pred.masks is not None and len(pred.masks) > 0:
                for i in range(len(pred.masks)):
                    # Redimensionner le masque à la taille de l'image originale
                    mask_tensor = pred.masks.data[i].cpu().numpy()
                    mask_uint8 = (mask_tensor * 255).astype(np.uint8)
                    if mask_uint8.shape[:2] != img.shape[:2]:
                        mask_uint8 = cv2.resize(mask_uint8, (img.shape[1], img.shape[0]),
                                                interpolation=cv2.INTER_NEAREST)
                    masks_list.append(mask_uint8)
                    scores_list.append(float(pred.boxes.conf[i].cpu().numpy()))

            print(f"    Fissures détectées : {len(masks_list)}")

            # Analyse morphologique
            if not args.no_analysis and masks_list:
                frame = analyze_frame(img, masks_list, scores_list, str(img_path), args.px_to_mm)
                print_analysis_summary(frame)
                annotated = draw_analysis(img, frame, masks_list)
                report_path = output_dir / "reports" / f"{img_path.stem}_analysis.json"
                save_analysis_report(frame, str(report_path))
                results_all.append(frame)
            else:
                annotated = pred.plot(masks=True, conf=True)

            # Sauvegarde
            out_img = output_dir / "annotated" / img_path.name
            out_img.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_img), annotated)
            print(f"    Sauvegardé : {out_img}")

            if args.save_masks:
                for j, mask in enumerate(masks_list):
                    mask_out = output_dir / "masks" / f"{img_path.stem}_mask_{j}.png"
                    mask_out.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(mask_out), mask)

    return results_all


# ─── Inférence Mask R-CNN ─────────────────────────────────────────────────────

def run_maskrcnn_inference(args, sources, output_dir: Path):
    try:
        from detectron2.engine import DefaultPredictor
        from detectron2.config import get_cfg
        from detectron2 import model_zoo
    except ImportError:
        sys.exit("❌ Detectron2 non installé. Voir README pour les instructions d'installation.")

    from scripts.crack_analysis import analyze_frame, draw_analysis, save_analysis_report, print_analysis_summary

    # Reconstruction minimale de la config pour le predictor
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score_thresh
    cfg.MODEL.WEIGHTS = args.model
    # Auto-détection GPU si --device non spécifié (utile sur Colab)
    if args.device is not None:
        cfg.MODEL.DEVICE = "cpu" if args.device == "cpu" else "cuda"
    else:
        try:
            import torch
            cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            cfg.MODEL.DEVICE = "cpu"
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333
    cfg.freeze()

    print(f"\n▶ Chargement du modèle Mask R-CNN : {args.model}")
    predictor = DefaultPredictor(cfg)

    results_all = []

    for img_path in sources:
        print(f"\n  Traitement : {img_path.name}")
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  ⚠ Impossible de lire : {img_path}")
            continue

        outputs = predictor(img)
        instances = outputs["instances"].to("cpu")
        masks_raw  = instances.pred_masks.numpy() if instances.has("pred_masks")  else []
        scores_raw = instances.scores.numpy()     if instances.has("scores")      else []

        masks_list  = [(m * 255).astype(np.uint8) for m in masks_raw]
        scores_list = [float(s) for s in scores_raw]

        print(f"    Fissures détectées : {len(masks_list)}")

        if not args.no_analysis and masks_list:
            frame = analyze_frame(img, masks_list, scores_list, str(img_path), args.px_to_mm)
            print_analysis_summary(frame)
            annotated = draw_analysis(img, frame, masks_list)
            report_path = output_dir / "reports" / f"{img_path.stem}_analysis.json"
            save_analysis_report(frame, str(report_path))
            results_all.append(frame)
        else:
            annotated = img.copy()

        out_img = output_dir / "annotated" / img_path.name
        out_img.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_img), annotated)
        print(f"    Sauvegardé : {out_img}")

        if args.save_masks:
            for j, mask in enumerate(masks_list):
                mask_out = output_dir / "masks" / f"{img_path.stem}_mask_{j}.png"
                mask_out.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(mask_out), mask)

    return results_all


# ─── Point d'entrée ───────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Détection du type de modèle
    if args.model_type is None:
        args.model_type = detect_model_type(args.model)
    print(f"\n🔍 Type de modèle : {args.model_type.upper()}")

    # Vérification du modèle
    if not Path(args.model).exists():
        sys.exit(f"❌ Modèle introuvable : {args.model}")

    # Collecte des sources
    sources, src_type = collect_sources(args.source)
    print(f"📂 Sources : {len(sources)} {src_type}")

    # Dossier de sortie
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("  Inférence — Détection de fissures")
    print("="*60)

    if args.model_type == "yolo":
        results = run_yolo_inference(args, sources, output_dir)
    else:
        results = run_maskrcnn_inference(args, sources, output_dir)

    print("\n" + "="*60)
    print(f"  ✓ Inférence terminée — {len(sources)} image(s) traitée(s)")
    print(f"  Résultats dans : {output_dir}/")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Ajouter le répertoire racine au PYTHONPATH
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    main()
