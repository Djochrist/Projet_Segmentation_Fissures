"""
Script d'entraînement Mask R-CNN - Fine-tuning Fissures (Detectron2)
Exécutable directement sur Google Colab ou en local.
"""

import os
import sys
import json
import shutil
import argparse
import yaml
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime


# ─── Helpers ──────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning Mask R-CNN pour la détection de fissures")
    parser.add_argument("--data-root", type=str, required=True,
                        help="Chemin vers le dossier du dataset COCO (contenant train/valid/test), "
                             "ou vers son dossier parent")
    parser.add_argument("--murs-sains", type=str, default=None,
                        help="Chemin vers le dossier d'images de murs sains (exemples négatifs, optionnel)")
    parser.add_argument("--murs-sains-masques", type=str, default=None,
                        help="Chemin vers le dossier de masques noirs des murs sains (optionnel, non requis)")
    parser.add_argument("--config", type=str, default="configs/maskrcnn_config.yaml",
                        help="Fichier de configuration YAML")
    parser.add_argument("--resume", action="store_true",
                        help="Reprendre depuis le dernier checkpoint dans OUTPUT_DIR")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Chemin explicite vers un checkpoint .pth")
    parser.add_argument("--max-iter", type=int, default=None,
                        help="Nombre d'itérations (surcharge la config)")
    parser.add_argument("--batch", type=int, default=None,
                        help="Images par batch (surcharge la config)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate de base (surcharge la config)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Dossier de sortie (surcharge la config)")
    return parser.parse_args()


def import_detectron2():
    """Importe Detectron2 et échoue proprement si absent."""
    try:
        import detectron2
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultTrainer, DefaultPredictor
        from detectron2.data import DatasetCatalog, MetadataCatalog
        from detectron2.data.datasets import register_coco_instances
        from detectron2.evaluation import COCOEvaluator
        from detectron2.utils.logger import setup_logger
        from detectron2 import model_zoo
        return True
    except ImportError:
        sys.exit(
            "❌ Detectron2 non installé.\n"
            "Sur Colab (GPU) :\n"
            "  !pip install 'git+https://github.com/facebookresearch/detectron2.git'\n"
        )


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def find_coco_dataset_dir(data_root: Path) -> Path:
    """
    Localise le dossier du dataset COCO-segmentation de façon robuste.

    Accepte que --data-root pointe :
      1) directement sur le dossier du dataset (contient train/valid/test
         avec _annotations.coco.json) ;
      2) sur un dossier parent contenant un sous-dossier nommé
         'segmentation_fissures.v6i.coco-segmentation' (ancienne convention) ;
      3) sur un dossier parent contenant n'importe quel sous-dossier avec un
         split 'train/_annotations.coco.json' (détection automatique).
    """
    if not data_root.exists():
        raise FileNotFoundError(
            f"Dossier introuvable : {data_root}\n"
            "Vérifiez le chemin passé à --data-root (le dossier doit être monté/synchronisé depuis Drive)."
        )

    def has_coco_structure(p: Path) -> bool:
        return (p / "train" / "_annotations.coco.json").exists()

    if has_coco_structure(data_root):
        return data_root

    legacy = data_root / "segmentation_fissures.v6i.coco-segmentation"
    if has_coco_structure(legacy):
        return legacy

    candidates = [d for d in data_root.iterdir() if d.is_dir() and has_coco_structure(d)]
    if len(candidates) == 1:
        print(f"  ℹ Dataset détecté automatiquement : {candidates[0].name}")
        return candidates[0]
    if len(candidates) > 1:
        raise FileNotFoundError(
            f"Plusieurs dossiers candidats trouvés sous {data_root} : "
            f"{[c.name for c in candidates]}. Précisez --data-root directement sur le bon dossier."
        )

    existing = [d.name for d in data_root.iterdir()] if data_root.is_dir() else []
    raise FileNotFoundError(
        f"Dataset COCO introuvable sous : {data_root}\n"
        f"Contenu actuel de ce dossier : {existing}\n"
        "Pointez --data-root directement sur le dossier du dataset Roboflow (export COCO Segmentation) "
        "qui contient train/, valid/, test/ avec un fichier _annotations.coco.json dans chacun."
    )


def check_coco_dataset(data_root: Path) -> dict:
    """Vérifie la structure du dataset COCO-segmentation."""
    dataset_dir = find_coco_dataset_dir(data_root)
    print(f"  Dataset COCO : {dataset_dir}")

    splits = {}
    for split in ("train", "valid", "test"):
        ann_file = dataset_dir / split / "_annotations.coco.json"
        img_dir  = dataset_dir / split
        if ann_file.exists():
            with open(ann_file) as f:
                coco = json.load(f)
            n_imgs = len(coco.get("images", []))
            n_anns = len(coco.get("annotations", []))
            splits[split] = {
                "annotation_file": str(ann_file),
                "image_dir": str(img_dir),
                "n_images": n_imgs,
                "n_annotations": n_anns,
            }
            print(f"  [{split:5s}] {n_imgs} images, {n_anns} annotations — {img_dir}")
        else:
            print(f"  [{split:5s}] ABSENT (ignoré)")

    if "train" not in splits:
        raise FileNotFoundError("Le split 'train' est obligatoire.")
    return splits, dataset_dir


def add_healthy_walls_coco(healthy_img_dir: Path, splits: dict, dataset_dir: Path) -> dict:
    """
    Intègre les murs sains dans le dataset COCO.
    Chaque image saine génère une entrée sans annotation (classe absente).
    Les masques noirs ne sont pas nécessaires : une entrée d'image sans
    annotation associée suffit pour signaler un mur sain.
    """
    if healthy_img_dir is None:
        print("  [murs_sains] Non fourni (--murs-sains), étape ignorée.")
        return splits

    if not healthy_img_dir.exists():
        print(f"  [murs_sains] ⚠ Dossier introuvable : {healthy_img_dir} — étape ignorée.")
        return splits

    img_exts = {".jpg", ".jpeg", ".png"}
    if (healthy_img_dir / "train").exists():
        search_dir = healthy_img_dir / "train"
    else:
        search_dir = healthy_img_dir
    images = [p for p in search_dir.iterdir() if p.is_file() and p.suffix.lower() in img_exts]

    if not images:
        print(f"  [murs_sains] ⚠ Aucune image trouvée dans {search_dir} — étape ignorée.")
        return splits
    print(f"  [murs_sains] ✓ {len(images)} image(s) trouvée(s)")

    # Charger l'annotation COCO du split train
    ann_path = Path(splits["train"]["annotation_file"])
    with open(ann_path) as f:
        coco_data = json.load(f)

    max_img_id = max((img["id"] for img in coco_data["images"]), default=0)
    train_img_dir = Path(splits["train"]["image_dir"])

    # Ensemble des noms de fichiers déjà présents dans le JSON (pour dédupliquer)
    existing_filenames = {img["file_name"] for img in coco_data["images"]}
    added = 0

    for img_path in images:
        # Éviter les doublons si le script est relancé
        if img_path.name in existing_filenames:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        new_id = max_img_id + added + 1

        # Copier l'image dans le dossier train
        dest = train_img_dir / img_path.name
        if not dest.exists():
            shutil.copy2(img_path, dest)

        # Ajouter l'entrée dans le JSON COCO (sans annotation = mur sain)
        coco_data["images"].append({
            "id": new_id,
            "file_name": img_path.name,
            "height": h,
            "width": w,
        })
        existing_filenames.add(img_path.name)
        added += 1

    # Sauvegarder le JSON enrichi
    with open(ann_path, "w") as f:
        json.dump(coco_data, f)
    print(f"  [murs_sains] ✓ {added} images saines ajoutées au JSON COCO")
    splits["train"]["n_images"] += added
    return splits


def get_thing_classes(annotation_file: str) -> list:
    """
    Lit les classes réelles depuis le fichier COCO (par id croissant), au lieu
    de supposer un nom de classe fixe ("fissure"). Les exports Roboflow COCO
    contiennent parfois une catégorie "supercategory" (id 0) en plus de la
    vraie classe (ex: ['segmentation-fissures', 'crack']) — on les garde
    toutes pour rester cohérent avec les annotations réellement présentes
    dans le JSON, quel que soit le nom donné par l'export.
    """
    with open(annotation_file) as f:
        coco = json.load(f)
    categories = sorted(coco.get("categories", []), key=lambda c: c["id"])
    return [c["name"] for c in categories]


def register_datasets(splits: dict):
    """Enregistre les datasets dans le DatasetCatalog de Detectron2."""
    from detectron2.data.datasets import register_coco_instances
    from detectron2.data import MetadataCatalog

    mapping = {"train": "fissures_train", "valid": "fissures_val", "test": "fissures_test"}
    thing_classes = get_thing_classes(splits["train"]["annotation_file"])
    print(f"  Classes détectées dans le JSON COCO : {thing_classes}")
    for split_key, catalog_name in mapping.items():
        if split_key in splits:
            s = splits[split_key]
            # Éviter les doublons si le script est réexécuté
            try:
                register_coco_instances(
                    catalog_name,
                    {"thing_classes": thing_classes},
                    s["annotation_file"],
                    s["image_dir"],
                )
            except AssertionError:
                pass  # Déjà enregistré
            print(f"  Dataset enregistré : {catalog_name}")
    return thing_classes


def build_cfg(args, yaml_cfg: dict, num_classes: int):
    """Construit la configuration Detectron2 à partir du YAML."""
    from detectron2.config import get_cfg
    from detectron2 import model_zoo

    cfg = get_cfg()

    # Charger un modèle de base Detectron2
    base_config = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(base_config))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(base_config)

    # Appliquer notre YAML
    # NUM_CLASSES doit correspondre exactement au nombre de classes réellement
    # présentes dans le JSON COCO (voir get_thing_classes) — sinon Detectron2
    # lève une erreur au moment de l'évaluation/entraînement. On ignore donc
    # toute valeur fixe du YAML pour éviter un décalage avec les données.
    model_cfg = yaml_cfg.get("MODEL", {})
    cfg.MODEL.ROI_HEADS.NUM_CLASSES         = num_classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST   = model_cfg.get("ROI_HEADS", {}).get("SCORE_THRESH_TEST", 0.5)
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST     = model_cfg.get("ROI_HEADS", {}).get("NMS_THRESH_TEST", 0.5)
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.25, 0.5, 1.0, 2.0, 4.0]]
    cfg.MODEL.BACKBONE.FREEZE_AT            = model_cfg.get("BACKBONE", {}).get("FREEZE_AT", 2)

    # Datasets
    cfg.DATASETS.TRAIN = ("fissures_train",)
    # Activer l'évaluation uniquement si le split valid est enregistré
    from detectron2.data import DatasetCatalog
    cfg.DATASETS.TEST  = ("fissures_val",) if "fissures_val" in DatasetCatalog.list() else ()

    # Dataloader
    dl_cfg = yaml_cfg.get("DATALOADER", {})
    cfg.DATALOADER.NUM_WORKERS              = dl_cfg.get("NUM_WORKERS", 4)
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = dl_cfg.get("FILTER_EMPTY_ANNOTATIONS", True)

    # Solver
    solver = yaml_cfg.get("SOLVER", {})
    cfg.SOLVER.BASE_LR          = args.lr       if args.lr       else solver.get("BASE_LR", 0.0001)
    cfg.SOLVER.MAX_ITER         = args.max_iter if args.max_iter else solver.get("MAX_ITER", 10000)
    cfg.SOLVER.STEPS            = tuple(solver.get("STEPS", [7000, 9000]))
    cfg.SOLVER.GAMMA            = solver.get("GAMMA", 0.1)
    cfg.SOLVER.WARMUP_ITERS     = solver.get("WARMUP_ITERS", 500)
    cfg.SOLVER.WARMUP_FACTOR    = solver.get("WARMUP_FACTOR", 0.001)
    cfg.SOLVER.CHECKPOINT_PERIOD= solver.get("CHECKPOINT_PERIOD", 500)
    cfg.SOLVER.IMS_PER_BATCH    = args.batch    if args.batch    else solver.get("IMS_PER_BATCH", 4)
    cfg.SOLVER.WEIGHT_DECAY     = solver.get("WEIGHT_DECAY", 0.0001)
    cfg.SOLVER.MOMENTUM         = solver.get("MOMENTUM", 0.9)
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED    = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
    cfg.SOLVER.AMP.ENABLED      = solver.get("AMP", {}).get("ENABLED", True)

    # Test
    test_cfg = yaml_cfg.get("TEST", {})
    cfg.TEST.EVAL_PERIOD        = test_cfg.get("EVAL_PERIOD", 500)
    cfg.TEST.DETECTIONS_PER_IMAGE = test_cfg.get("DETECTIONS_PER_IMAGE", 100)

    # Input / augmentation
    input_cfg = yaml_cfg.get("INPUT", {})
    cfg.INPUT.MIN_SIZE_TRAIN    = (480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800)
    cfg.INPUT.MAX_SIZE_TRAIN    = input_cfg.get("MAX_SIZE_TRAIN", 1333)
    cfg.INPUT.MIN_SIZE_TEST     = input_cfg.get("MIN_SIZE_TEST", 800)
    cfg.INPUT.MAX_SIZE_TEST     = input_cfg.get("MAX_SIZE_TEST", 1333)
    cfg.INPUT.RANDOM_FLIP       = input_cfg.get("RANDOM_FLIP", "horizontal")
    # Les exports Roboflow COCO mélangent parfois plusieurs représentations de
    # segmentation (polygones imbriqués différemment, RLE...) selon les
    # instances. Le mode "polygon" de Detectron2 est strict et plante dès
    # qu'une annotation ne correspond pas exactement au format attendu
    # (ValueError: "Expect a list of polygons per instance. Got ndarray").
    # Le mode "bitmask" est tolérant à tous ces formats (polygones, RLE,
    # masques bruts) et est utilisé ici pour éviter ce plantage.
    cfg.INPUT.MASK_FORMAT       = "bitmask"

    # Sortie
    output_dir = args.output_dir if args.output_dir else yaml_cfg.get("OUTPUT_DIR", "outputs/maskrcnn/run")
    cfg.OUTPUT_DIR = output_dir

    # Reprise
    if args.resume_from:
        cfg.MODEL.WEIGHTS = args.resume_from

    cfg.SEED = yaml_cfg.get("SEED", 42)
    cfg.freeze()
    return cfg


# ─── Trainer personnalisé ─────────────────────────────────────────────────────

class FissureTrainer:
    """Wrapper autour de DefaultTrainer avec évaluation COCO à chaque période."""

    def __init__(self, cfg):
        from detectron2.engine import DefaultTrainer
        from detectron2.evaluation import COCOEvaluator

        class _Trainer(DefaultTrainer):
            @classmethod
            def build_evaluator(cls, cfg, dataset_name, output_folder=None):
                if output_folder is None:
                    output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
                return COCOEvaluator(dataset_name, cfg, True, output_folder)

        self.TrainerCls = _Trainer
        self.cfg = cfg

    def train(self, resume: bool = False):
        from detectron2.utils.logger import setup_logger
        setup_logger()
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        trainer = self.TrainerCls(self.cfg)
        trainer.resume_or_load(resume=resume)
        return trainer.train()


# ─── Entraînement ─────────────────────────────────────────────────────────────

def train(args):
    import_detectron2()

    print("\n" + "="*60)
    print("  Fine-tuning Mask R-CNN — Détection de fissures")
    print("="*60 + "\n")

    yaml_cfg = load_config(args.config)
    data_root = Path(args.data_root)

    print("📂 Vérification du dataset COCO...")
    splits, dataset_dir = check_coco_dataset(data_root)

    print("🌿 Intégration des murs sains...")
    healthy_dir = Path(args.murs_sains) if args.murs_sains else None
    splits = add_healthy_walls_coco(healthy_dir, splits, dataset_dir)

    print("📝 Enregistrement des datasets Detectron2...")
    thing_classes = register_datasets(splits)

    print("⚙ Construction de la configuration...")
    cfg = build_cfg(args, yaml_cfg, num_classes=len(thing_classes))

    print(f"\n⚙ Hyperparamètres principaux :")
    print(f"   Max iter   : {cfg.SOLVER.MAX_ITER}")
    print(f"   LR         : {cfg.SOLVER.BASE_LR}")
    print(f"   Batch      : {cfg.SOLVER.IMS_PER_BATCH}")
    print(f"   Eval period: {cfg.TEST.EVAL_PERIOD}")
    print(f"   AMP        : {cfg.SOLVER.AMP.ENABLED}")
    print(f"   Output dir : {cfg.OUTPUT_DIR}")

    resume = args.resume or (args.resume_from is not None)
    print(f"\n🚀 Démarrage de l'entraînement (resume={resume})...\n")

    trainer = FissureTrainer(cfg)
    trainer.train(resume=resume)

    print("\n" + "="*60)
    print("  Entraînement terminé")
    print("="*60)
    print(f"  Modèles sauvegardés dans : {cfg.OUTPUT_DIR}")
    print("  Meilleur modèle          : model_best.pth (si eval_period actif)")
    print("  Dernier modèle           : model_final.pth")
    print("\n  Pour reprendre :")
    print(f"  python scripts/train_maskrcnn.py --data-root <DATA> --resume")


# ─── Point d'entrée ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    train(args)
