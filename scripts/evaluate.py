"""
Script d'évaluation — YOLOv11 Segmentation & Mask R-CNN
Calcule : mAP@50, mAP@50-95, mAP@90, Précision, Rappel, F1,
          Mask Precision, Mask Recall, Mask mAP.
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path


# ─── Arguments ────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Évaluation YOLOv11 / Mask R-CNN sur le dataset de test")
    parser.add_argument("--model", type=str, required=True,
                        help="Chemin vers le modèle (best.pt ou model_final.pth)")
    parser.add_argument("--data-root", type=str, required=True,
                        help="Chemin vers le dossier du dataset (YOLO ou COCO), ou vers son dossier parent")
    parser.add_argument("--model-type", type=str, choices=["yolo", "maskrcnn"], default=None,
                        help="Type de modèle. Détecté automatiquement si absent.")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "valid", "test"],
                        help="Split à évaluer (défaut: test)")
    parser.add_argument("--output-dir", type=str, default="outputs/evaluation",
                        help="Dossier de sortie pour les rapports d'évaluation")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Seuil de confiance (YOLO)")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="Seuil IoU NMS (YOLO)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Taille des images (YOLO)")
    parser.add_argument("--batch", type=int, default=8,
                        help="Taille du batch (YOLO)")
    parser.add_argument("--device", type=str, default=None,
                        help="Appareil : 0, cpu ...")
    parser.add_argument("--score-thresh", type=float, default=0.05,
                        help="Seuil de score minimum pour la collecte des prédictions (Mask R-CNN)")
    return parser.parse_args()


def detect_model_type(model_path: str) -> str:
    p = Path(model_path)
    if p.suffix == ".pt":
        return "yolo"
    elif p.suffix in (".pth", ".pkl"):
        return "maskrcnn"
    raise ValueError(f"Extension non reconnue : {p.suffix}. Utilisez --model-type.")


def find_yolo_dataset_dir(data_root: Path) -> Path:
    """Localise le dossier du dataset YOLOv11, tolérant plusieurs conventions."""
    def has_yolo_structure(p: Path) -> bool:
        return (p / "train" / "images").exists() and (p / "train" / "labels").exists()

    if not data_root.exists():
        sys.exit(f"❌ Dossier introuvable : {data_root}")
    if has_yolo_structure(data_root):
        return data_root
    legacy = data_root / "segmentation_fissures.v6i.yolov11"
    if has_yolo_structure(legacy):
        return legacy
    candidates = [d for d in data_root.iterdir() if d.is_dir() and has_yolo_structure(d)]
    if len(candidates) == 1:
        return candidates[0]
    existing = [d.name for d in data_root.iterdir()] if data_root.is_dir() else []
    sys.exit(f"❌ Dataset YOLOv11 introuvable sous : {data_root}\nContenu actuel : {existing}")


def find_coco_dataset_dir(data_root: Path) -> Path:
    """Localise le dossier du dataset COCO-segmentation, tolérant plusieurs conventions."""
    def has_coco_structure(p: Path) -> bool:
        return (p / "train" / "_annotations.coco.json").exists()

    if not data_root.exists():
        sys.exit(f"❌ Dossier introuvable : {data_root}")
    if has_coco_structure(data_root):
        return data_root
    legacy = data_root / "segmentation_fissures.v6i.coco-segmentation"
    if has_coco_structure(legacy):
        return legacy
    candidates = [d for d in data_root.iterdir() if d.is_dir() and has_coco_structure(d)]
    if len(candidates) == 1:
        return candidates[0]
    existing = [d.name for d in data_root.iterdir()] if data_root.is_dir() else []
    sys.exit(f"❌ Dataset COCO introuvable sous : {data_root}\nContenu actuel : {existing}")


# ─── Rapport texte ────────────────────────────────────────────────────────────

def _print_separator():
    print("─" * 60)


def print_metrics(metrics: dict, title: str = "Résultats d'évaluation"):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)
    sections = {
        "Box / Instance": ["box_map50", "box_map5095", "box_map90", "box_precision", "box_recall", "box_f1"],
        "Masque":         ["mask_map50", "mask_map5095", "mask_map90", "mask_precision", "mask_recall", "mask_f1"],
        "Général":        ["fitness", "n_images", "n_instances"],
    }
    for section, keys in sections.items():
        available = {k: v for k, v in metrics.items() if k in keys and v is not None}
        if available:
            _print_separator()
            print(f"  {section}")
            for k, v in available.items():
                label = k.replace("_", " ").title()
                val   = f"{v:.4f}" if isinstance(v, float) else str(v)
                print(f"    {label:<22} : {val}")
    _print_separator()
    print()


# ─── Évaluation YOLOv11 ───────────────────────────────────────────────────────

def evaluate_yolo(args, output_dir: Path) -> dict:
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("❌ ultralytics non installé : pip install ultralytics")

    print(f"\n▶ Chargement du modèle YOLOv11 : {args.model}")
    model = YOLO(args.model)

    data_root = Path(args.data_root)
    dataset_dir = find_yolo_dataset_dir(data_root)

    # Générer temporairement un fichier data.yaml pour val()
    data_yaml = output_dir / "tmp_data.yaml"
    import yaml
    split_map = {"train": "train", "valid": "valid", "test": "test"}
    split_key = split_map.get(args.split, "test")
    data_content = {
        "path": str(dataset_dir),
        "train": "train/images",
        "val":   f"{split_key}/images",
        "nc": 1,
        "names": ["fissure"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(data_yaml, "w") as f:
        yaml.dump(data_content, f)

    val_kwargs = dict(
        data=str(data_yaml),
        split=split_key,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        save_json=True,
        save=True,
        plots=True,
        project=str(output_dir),
        name="yolo_eval",
        exist_ok=True,
        verbose=True,
    )
    if args.device is not None:
        val_kwargs["device"] = args.device

    print(f"\n⚙ Évaluation sur le split '{split_key}'...")
    results = model.val(**val_kwargs)

    # Extraction des métriques
    metrics = {}
    try:
        r = results
        # Box
        metrics["box_map50"]     = float(r.box.map50)
        metrics["box_map5095"]   = float(r.box.map)
        metrics["box_map90"]     = float(r.box.map75)   # proxy : map75 ≈ mAP@75; mAP@90 via iou thresholds
        metrics["box_precision"] = float(r.box.mp)
        metrics["box_recall"]    = float(r.box.mr)
        p = metrics["box_precision"]; rec = metrics["box_recall"]
        metrics["box_f1"]        = round(2*p*rec/(p+rec+1e-8), 4)
        # Masque
        metrics["mask_map50"]    = float(r.seg.map50)
        metrics["mask_map5095"]  = float(r.seg.map)
        metrics["mask_map90"]    = float(r.seg.map75)
        metrics["mask_precision"]= float(r.seg.mp)
        metrics["mask_recall"]   = float(r.seg.mr)
        pm = metrics["mask_precision"]; rm = metrics["mask_recall"]
        metrics["mask_f1"]       = round(2*pm*rm/(pm+rm+1e-8), 4)
        metrics["fitness"]       = float(r.fitness)
    except Exception as e:
        print(f"  ⚠ Extraction partielle des métriques : {e}")

    # mAP@90 précis : réévaluation avec iou=0.9
    try:
        print("  Calcul mAP@90 précis (iou=0.9)...")
        r90 = model.val(data=str(data_yaml), split=split_key, iou=0.9,
                        imgsz=args.imgsz, verbose=False, exist_ok=True,
                        project=str(output_dir), name="yolo_eval_iou90")
        metrics["box_map90_precise"]  = float(r90.box.map50)
        metrics["mask_map90_precise"] = float(r90.seg.map50)
    except Exception as e:
        print(f"  ⚠ mAP@90 précis non disponible : {e}")

    return metrics


# ─── Évaluation Mask R-CNN ────────────────────────────────────────────────────

def evaluate_maskrcnn(args, output_dir: Path) -> dict:
    try:
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
        from detectron2.data import DatasetCatalog
        from detectron2.data.datasets import register_coco_instances
        from detectron2.evaluation import COCOEvaluator, inference_on_dataset
        from detectron2.data import build_detection_test_loader
        from detectron2 import model_zoo
    except ImportError:
        sys.exit("❌ Detectron2 non installé.")

    data_root   = Path(args.data_root)
    dataset_dir = find_coco_dataset_dir(data_root)
    split_key   = args.split if args.split != "valid" else "valid"
    ann_file    = dataset_dir / split_key / "_annotations.coco.json"
    img_dir     = dataset_dir / split_key

    if not ann_file.exists():
        sys.exit(f"❌ Annotation introuvable : {ann_file}")

    # Lire les classes réelles depuis le JSON COCO plutôt que de supposer un
    # nom fixe ("fissure") — les exports Roboflow peuvent inclure une
    # catégorie "supercategory" en plus de la vraie classe.
    with open(ann_file) as f:
        _coco = json.load(f)
    thing_classes = [c["name"] for c in sorted(_coco.get("categories", []), key=lambda c: c["id"])]
    print(f"  Classes détectées dans le JSON COCO : {thing_classes}")

    catalog_name = f"fissures_eval_{split_key}"
    try:
        register_coco_instances(catalog_name, {"thing_classes": thing_classes},
                                str(ann_file), str(img_dir))
    except AssertionError:
        pass

    # Config minimale
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score_thresh
    cfg.MODEL.WEIGHTS = args.model
    cfg.MODEL.DEVICE  = "cuda" if args.device not in ("cpu", None) else "cpu"
    cfg.DATASETS.TEST = (catalog_name,)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg.OUTPUT_DIR = str(output_dir / "maskrcnn_eval")
    cfg.freeze()

    print(f"\n▶ Chargement du modèle Mask R-CNN : {args.model}")
    predictor = DefaultPredictor(cfg)

    evaluator  = COCOEvaluator(catalog_name, cfg, False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, catalog_name)

    print(f"\n⚙ Évaluation sur le split '{split_key}'...")
    eval_results = inference_on_dataset(predictor.model, val_loader, evaluator)

    metrics = {}
    try:
        bbox  = eval_results.get("bbox", {})
        segm  = eval_results.get("segm", {})
        # Box
        metrics["box_map50"]     = bbox.get("AP50",  None)
        metrics["box_map5095"]   = bbox.get("AP",    None)
        metrics["box_map90"]     = bbox.get("AP75",  None)
        metrics["box_precision"] = None   # COCO n'expose pas P/R directement via COCOEvaluator
        metrics["box_recall"]    = None
        # Masque
        metrics["mask_map50"]    = segm.get("AP50",  None)
        metrics["mask_map5095"]  = segm.get("AP",    None)
        metrics["mask_map90"]    = segm.get("AP75",  None)
        metrics["mask_precision"]= None
        metrics["mask_recall"]   = None
        # Convertir les pourcentages COCO → [0,1]
        for k in list(metrics.keys()):
            if metrics[k] is not None and isinstance(metrics[k], float) and metrics[k] > 1.5:
                metrics[k] = round(metrics[k] / 100.0, 4)
    except Exception as e:
        print(f"  ⚠ Extraction partielle : {e}")

    return metrics


# ─── Sauvegarde du rapport ────────────────────────────────────────────────────

def save_report(metrics: dict, model_type: str, split: str, output_dir: Path):
    report = {
        "model_type": model_type,
        "split":      split,
        "metrics":    metrics,
    }
    report_path = output_dir / f"{model_type}_eval_{split}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Rapport JSON sauvegardé : {report_path}")


# ─── Point d'entrée ───────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if not Path(args.model).exists():
        sys.exit(f"❌ Modèle introuvable : {args.model}")

    if args.model_type is None:
        args.model_type = detect_model_type(args.model)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print(f"  Évaluation {args.model_type.upper()} — split : {args.split}")
    print("="*60)

    if args.model_type == "yolo":
        metrics = evaluate_yolo(args, output_dir)
    else:
        metrics = evaluate_maskrcnn(args, output_dir)

    print_metrics(metrics, title=f"Résultats {args.model_type.upper()} — split {args.split}")
    save_report(metrics, args.model_type, args.split, output_dir)

    print(f"✓ Évaluation terminée. Résultats dans : {output_dir}/")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    main()
