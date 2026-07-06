import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--model-type", type=str, choices=["yolo", "maskrcnn"], default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--output-dir", type=str, default="outputs/evaluation")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--score-thresh", type=float, default=0.05)
    return parser.parse_args()


def detect_model_type(model_path: str) -> str:
    p = Path(model_path)
    if p.suffix == ".pt":
        return "yolo"
    elif p.suffix in (".pth", ".pkl"):
        return "maskrcnn"
    raise ValueError(f"Extension non reconnue : {p.suffix}. Utilisez --model-type.")


def find_yolo_dataset_dir(data_root: Path) -> Path:
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
    sys.exit(f"❌ Dataset YOLOv11 introuvable sous : {data_root}\nContenu : {existing}")


def find_coco_dataset_dir(data_root: Path) -> Path:
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
    sys.exit(f"❌ Dataset COCO introuvable sous : {data_root}\nContenu : {existing}")


def _print_separator():
    print("─" * 60)


def _compute_precision_recall_f1(coco_gt, dt_json_path: str, iou_type: str,
                                  score_thresh: float = 0.5, iou_thr: float = 0.5):
    """Calcule précision / rappel / F1 à un seuil de confiance et d'IoU donnés.

    On réutilise directement les tableaux `precision`/`recall`/`scores` que
    pycocotools calcule lui-même dans `COCOeval.accumulate()` (méthode
    officielle, la même que celle utilisée pour l'AP/AR), plutôt que de
    recompter les TP/FP/FN à la main. Pour chaque catégorie, pycocotools
    donne, à chaque niveau de rappel R (101 points de 0 à 1), la précision
    obtenue et le score de confiance minimal nécessaire pour l'atteindre.
    On cherche donc le point de rappel le plus élevé encore accessible avec
    un seuil de confiance >= score_thresh, et on lit précision/rappel à cet
    endroit — pas de logique de comptage réinventée.
    """
    from pycocotools.cocoeval import COCOeval

    coco_dt = coco_gt.loadRes(dt_json_path)
    E = COCOeval(coco_gt, coco_dt, iou_type)
    E.params.iouThrs = np.array([iou_thr])
    E.params.areaRng = [[0, 1e5 ** 2]]
    E.params.areaRngLbl = ["all"]
    E.params.maxDets = [100]
    E.evaluate()
    E.accumulate()

    # Tableaux [T, R, K, A, M] = [iouThrs, recallThrs, catégories, areaRng, maxDets]
    # Ici T=A=M=1 (un seul seuil IoU, une seule plage d'aire, un seul maxDets).
    precision_arr = E.eval["precision"][0, :, :, 0, 0]  # [R, K]
    scores_arr = E.eval["scores"][0, :, :, 0, 0]        # [R, K]
    recall_thrs = E.params.recThrs                       # [R]

    n_cats = precision_arr.shape[1]
    precisions, recalls = [], []
    for k in range(n_cats):
        p_k = precision_arr[:, k]
        s_k = scores_arr[:, k]
        valid = p_k > -1
        if not np.any(valid):
            continue
        candidates = np.where(valid & (s_k >= score_thresh))[0]
        if len(candidates) == 0:
            precisions.append(0.0)
            recalls.append(0.0)
            continue
        best_idx = candidates[-1]  # rappel max atteignable à ce seuil de confiance
        precisions.append(float(p_k[best_idx]))
        recalls.append(float(recall_thrs[best_idx]))

    precision = float(np.mean(precisions)) if precisions else 0.0
    recall = float(np.mean(recalls)) if recalls else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def print_metrics(metrics: dict, title: str = "Résultats"):
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


def _normalize_metric_value(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if value > 1.5:
        return round(value / 100.0, 4)
    return round(value, 4)


def _compute_f1(precision, recall):
    if precision is None or recall is None:
        return None
    try:
        precision = float(precision)
        recall = float(recall)
    except (TypeError, ValueError):
        return None
    if precision + recall <= 0:
        return None
    return round(2 * precision * recall / (precision + recall + 1e-8), 4)


def normalize_maskrcnn_metrics(eval_results: dict) -> dict:
    bbox = eval_results.get("bbox", {})
    segm = eval_results.get("segm", {})
    metrics = {}
    metrics["box_map50"] = _normalize_metric_value(bbox.get("AP50"))
    metrics["box_map5095"] = _normalize_metric_value(bbox.get("AP"))
    metrics["box_map90"] = _normalize_metric_value(bbox.get("AP75"))
    metrics["box_precision"] = _normalize_metric_value(bbox.get("AP50"))
    metrics["box_recall"] = _normalize_metric_value(bbox.get("AR@100"))
    metrics["box_f1"] = _compute_f1(metrics["box_precision"], metrics["box_recall"])

    metrics["mask_map50"] = _normalize_metric_value(segm.get("AP50"))
    metrics["mask_map5095"] = _normalize_metric_value(segm.get("AP"))
    metrics["mask_map90"] = _normalize_metric_value(segm.get("AP75"))
    metrics["mask_precision"] = _normalize_metric_value(segm.get("AP50"))
    metrics["mask_recall"] = _normalize_metric_value(segm.get("AR@100"))
    metrics["mask_f1"] = _compute_f1(metrics["mask_precision"], metrics["mask_recall"])
    return metrics


def evaluate_yolo(args, output_dir: Path) -> dict:
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("❌ ultralytics non installé : pip install ultralytics")

    print(f"\n▶ Modèle YOLOv11 : {args.model}")
    model = YOLO(args.model)

    data_root = Path(args.data_root)
    dataset_dir = find_yolo_dataset_dir(data_root)

    data_yaml = output_dir / "tmp_data.yaml"
    import yaml
    split_key = args.split
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

    metrics = {}
    try:
        r = results
        metrics["box_map50"]     = float(r.box.map50)
        metrics["box_map5095"]   = float(r.box.map)
        metrics["box_map90"]     = float(r.box.map75)
        metrics["box_precision"] = float(r.box.mp)
        metrics["box_recall"]    = float(r.box.mr)
        p = metrics["box_precision"]; rec = metrics["box_recall"]
        metrics["box_f1"]        = round(2*p*rec/(p+rec+1e-8), 4)
        metrics["mask_map50"]    = float(r.seg.map50)
        metrics["mask_map5095"]  = float(r.seg.map)
        metrics["mask_map90"]    = float(r.seg.map75)
        metrics["mask_precision"]= float(r.seg.mp)
        metrics["mask_recall"]   = float(r.seg.mr)
        pm = metrics["mask_precision"]; rm = metrics["mask_recall"]
        metrics["mask_f1"]       = round(2*pm*rm/(pm+rm+1e-8), 4)
        metrics["fitness"]       = float(r.fitness)
    except Exception as e:
        print(f"  ⚠ Extraction partielle : {e}")

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
    split_key   = args.split
    ann_file    = dataset_dir / split_key / "_annotations.coco.json"
    img_dir     = dataset_dir / split_key

    if not ann_file.exists():
        sys.exit(f"❌ Annotation introuvable : {ann_file}")

    with open(ann_file) as f:
        _coco = json.load(f)
    _used_ids = {ann["category_id"] for ann in _coco.get("annotations", [])}
    thing_classes = [
        c["name"]
        for c in sorted(_coco.get("categories", []), key=lambda c: c["id"])
        if c["id"] in _used_ids
    ]
    print(f"  Classes réellement utilisées : {thing_classes}")

    catalog_name = f"fissures_eval_{split_key}"
    try:
        register_coco_instances(catalog_name, {"thing_classes": thing_classes},
                                str(ann_file), str(img_dir))
    except AssertionError:
        pass

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score_thresh
    cfg.MODEL.WEIGHTS = args.model
    cfg.MODEL.DEVICE  = "cuda" if args.device not in ("cpu", None) else "cpu"
    cfg.DATASETS.TEST = (catalog_name,)
    cfg.INPUT.MASK_FORMAT = "bitmask"
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg.OUTPUT_DIR = str(output_dir / "maskrcnn_eval")
    cfg.freeze()

    print(f"\n▶ Modèle Mask R-CNN : {args.model}")
    predictor = DefaultPredictor(cfg)

    evaluator  = COCOEvaluator(catalog_name, cfg, False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, catalog_name)

    print(f"\n⚙ Évaluation sur le split '{split_key}'...")
    eval_results = inference_on_dataset(predictor.model, val_loader, evaluator)

    metrics = {}
    try:
        metrics = normalize_maskrcnn_metrics(eval_results)
    except Exception as e:
        print(f"  ⚠ Extraction partielle (mAP) : {e}")

    try:
        dt_json = Path(cfg.OUTPUT_DIR) / "coco_instances_results.json"
        if dt_json.exists():
            from pycocotools.coco import COCO
            coco_gt = COCO(str(ann_file))
            score_thresh = args.score_thresh
            box_p, box_r, box_f1 = _compute_precision_recall_f1(
                coco_gt, str(dt_json), "bbox", score_thresh=score_thresh)
            metrics["box_precision"] = round(box_p, 4)
            metrics["box_recall"]    = round(box_r, 4)
            metrics["box_f1"]        = round(box_f1, 4)
            if segm:
                mask_p, mask_r, mask_f1 = _compute_precision_recall_f1(
                    coco_gt, str(dt_json), "segm", score_thresh=score_thresh)
                metrics["mask_precision"] = round(mask_p, 4)
                metrics["mask_recall"]    = round(mask_r, 4)
                metrics["mask_f1"]        = round(mask_f1, 4)
        else:
            print(f"  ⚠ Fichier de prédictions introuvable pour precision/recall : {dt_json}")
    except Exception as e:
        print(f"  ⚠ Extraction partielle (precision/recall/F1) : {e}")

    return metrics


def save_report(metrics: dict, model_type: str, split: str, output_dir: Path):
    report = {"model_type": model_type, "split": split, "metrics": metrics}
    report_path = output_dir / f"{model_type}_eval_{split}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Rapport JSON : {report_path}")


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
    print(f"✓ Évaluation terminée. Résultats : {output_dir}/")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    main()
