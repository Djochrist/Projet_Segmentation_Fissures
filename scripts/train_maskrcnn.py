import os
import sys
import json
import shutil
import argparse
import subprocess
import yaml
from pathlib import Path
from datetime import datetime

try:
    import cv2
except ImportError:  # pragma: no cover - dépendance optionnelle pour les tests
    cv2 = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--murs-sains", type=str, default=None)
    parser.add_argument("--murs-sains-masques", type=str, default=None)
    parser.add_argument("--config", type=str, default="configs/maskrcnn_config.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--eval-test", action="store_true",
                        help="Évaluer automatiquement le split test après l'entraînement")
    return parser.parse_args()


def import_detectron2():
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
            "Sur Colab : !pip install 'git+https://github.com/facebookresearch/detectron2.git'"
        )


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _drive_hint(data_root: Path) -> str:
    """Construit un message d'aide listant le contenu du Drive monté quand
    un chemin de dataset est introuvable, pour repérer rapidement le bon nom."""
    lines = []
    ancestor = data_root
    while not ancestor.exists() and ancestor.parent != ancestor:
        ancestor = ancestor.parent
    if ancestor.exists() and ancestor.is_dir():
        try:
            entries = sorted(p.name for p in ancestor.iterdir())
        except OSError:
            entries = []
        lines.append(f"  Contenu de {ancestor} : {entries}")

    mydrive = Path("/content/drive/MyDrive")
    if mydrive.exists() and mydrive != ancestor:
        try:
            entries = sorted(p.name for p in mydrive.iterdir())
        except OSError:
            entries = []
        lines.append(f"  Contenu de {mydrive} : {entries}")

    if not lines:
        lines.append("  (Google Drive ne semble pas monté : vérifie drive.mount('/content/drive'))")
    return "\n".join(lines)


def find_coco_dataset_dir(data_root: Path) -> Path:
    if not data_root.exists():
        raise FileNotFoundError(
            f"Dossier introuvable : {data_root}\n"
            f"Vérifie le chemin exact de ton dataset sur Drive :\n{_drive_hint(data_root)}"
        )

    def has_coco_structure(p: Path) -> bool:
        return (p / "train" / "_annotations.coco.json").exists()

    if has_coco_structure(data_root):
        return data_root

    for legacy_name in ("segmentation_fissures.v7i.coco-segmentation",
                         "segmentation_fissures.v6i.coco-segmentation"):
        legacy = data_root / legacy_name
        if has_coco_structure(legacy):
            return legacy

    candidates = [d for d in data_root.iterdir() if d.is_dir() and has_coco_structure(d)]
    if len(candidates) == 1:
        print(f"  Dataset détecté : {candidates[0].name}")
        return candidates[0]
    if len(candidates) > 1:
        raise FileNotFoundError(
            f"Plusieurs datasets trouvés sous {data_root} : {[c.name for c in candidates]}. "
            "Précisez --data-root directement."
        )

    existing = [d.name for d in data_root.iterdir()] if data_root.is_dir() else []
    raise FileNotFoundError(
        f"Dataset COCO introuvable sous : {data_root}\nContenu : {existing}\n"
        f"{_drive_hint(data_root)}"
    )


def check_coco_dataset(data_root: Path) -> dict:
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
            print(f"  [{split:5s}] {n_imgs} images, {n_anns} annotations")
        else:
            print(f"  [{split:5s}] ABSENT (ignoré)")
    if "train" not in splits:
        raise FileNotFoundError("Le split 'train' est obligatoire.")
    return splits, dataset_dir


def add_healthy_walls_coco(healthy_img_dir: Path, splits: dict, dataset_dir: Path) -> dict:
    if healthy_img_dir is None:
        print("  [murs_sains] Non fourni, étape ignorée.")
        return splits
    if not healthy_img_dir.exists():
        print(f"  [murs_sains] ⚠ Dossier introuvable : {healthy_img_dir} — ignoré.")
        return splits

    img_exts = {".jpg", ".jpeg", ".png"}
    search_dir = healthy_img_dir / "train" if (healthy_img_dir / "train").exists() else healthy_img_dir
    images = [p for p in search_dir.iterdir() if p.is_file() and p.suffix.lower() in img_exts]

    if not images:
        print(f"  [murs_sains] ⚠ Aucune image dans {search_dir} — ignoré.")
        return splits
    print(f"  [murs_sains] ✓ {len(images)} image(s) trouvée(s)")

    ann_path = Path(splits["train"]["annotation_file"])
    with open(ann_path) as f:
        coco_data = json.load(f)

    max_img_id = max((img["id"] for img in coco_data["images"]), default=0)
    train_img_dir = Path(splits["train"]["image_dir"])
    existing_filenames = {img["file_name"] for img in coco_data["images"]}
    added = 0

    for img_path in images:
        if img_path.name in existing_filenames:
            continue
        if cv2 is None:
            print(f"  [murs_sains] ⚠ cv2 non disponible ; image ignorée : {img_path}")
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        new_id = max_img_id + added + 1
        dest = train_img_dir / img_path.name
        if not dest.exists():
            shutil.copy2(img_path, dest)
        coco_data["images"].append({
            "id": new_id,
            "file_name": img_path.name,
            "height": h,
            "width": w,
        })
        existing_filenames.add(img_path.name)
        added += 1

    with open(ann_path, "w") as f:
        json.dump(coco_data, f)
    print(f"  [murs_sains] ✓ {added} images saines ajoutées")
    splits["train"]["n_images"] += added
    return splits


def get_thing_classes(annotation_file: str) -> list:
    with open(annotation_file) as f:
        coco = json.load(f)
    used_ids = {ann["category_id"] for ann in coco.get("annotations", [])}
    categories = sorted(
        [c for c in coco.get("categories", []) if c["id"] in used_ids],
        key=lambda c: c["id"]
    )
    classes = [c["name"] for c in categories]
    print(f"  Classes réellement utilisées : {classes}")
    return classes


def clean_coco_categories(annotation_file: str, reference_classes: list) -> None:
    """Réécrit la liste 'categories' du fichier COCO pour ne garder que les
    catégories réellement utilisées dans les annotations, alignées sur
    `reference_classes`. Indispensable car `load_coco_json` de Detectron2
    recalcule lui-même `thing_classes` à partir du JSON brut (catégories
    fantômes Roboflow incluses) à chaque chargement paresseux du dataset —
    sans ce nettoyage, cette valeur recalculée entre en conflit avec celle
    déjà enregistrée dans MetadataCatalog et lève une AssertionError.
    """
    with open(annotation_file) as f:
        coco = json.load(f)

    used_ids = {ann["category_id"] for ann in coco.get("annotations", [])}
    categories = sorted(
        [c for c in coco.get("categories", []) if c["id"] in used_ids],
        key=lambda c: c["id"]
    )

    if [c["name"] for c in categories] == list(reference_classes) and len(categories) == len(coco.get("categories", [])):
        return

    coco["categories"] = categories
    with open(annotation_file, "w") as f:
        json.dump(coco, f)


def register_datasets(splits: dict):
    from detectron2.data.datasets import register_coco_instances
    from detectron2.data import DatasetCatalog, MetadataCatalog

    mapping = {"train": "fissures_train", "valid": "fissures_val", "test": "fissures_test"}
    thing_classes = get_thing_classes(splits["train"]["annotation_file"])

    for split_key, catalog_name in mapping.items():
        if split_key in splits:
            s = splits[split_key]
            clean_coco_categories(s["annotation_file"], thing_classes)

            if catalog_name in DatasetCatalog.list():
                DatasetCatalog.remove(catalog_name)
            if catalog_name in MetadataCatalog:
                MetadataCatalog.remove(catalog_name)

            register_coco_instances(
                catalog_name,
                {"thing_classes": thing_classes},
                s["annotation_file"],
                s["image_dir"],
            )
            print(f"  Dataset enregistré : {catalog_name}")
    return thing_classes


def build_cfg(args, yaml_cfg: dict, num_classes: int):
    from detectron2.config import get_cfg
    from detectron2 import model_zoo

    cfg = get_cfg()
    base_config = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(base_config))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(base_config)

    model_cfg = yaml_cfg.get("MODEL", {})
    cfg.MODEL.ROI_HEADS.NUM_CLASSES         = num_classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST   = model_cfg.get("ROI_HEADS", {}).get("SCORE_THRESH_TEST", 0.5)
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST     = model_cfg.get("ROI_HEADS", {}).get("NMS_THRESH_TEST", 0.5)
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.25, 0.5, 1.0, 2.0, 4.0]]
    anchor_sizes = model_cfg.get("ANCHOR_GENERATOR", {}).get("SIZES")
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = anchor_sizes if anchor_sizes else [[16], [32], [64], [128], [256]]
    cfg.MODEL.BACKBONE.FREEZE_AT            = model_cfg.get("BACKBONE", {}).get("FREEZE_AT", 2)

    cfg.DATASETS.TRAIN = ("fissures_train",)
    from detectron2.data import DatasetCatalog
    cfg.DATASETS.TEST  = ("fissures_val",) if "fissures_val" in DatasetCatalog.list() else ()

    dl_cfg = yaml_cfg.get("DATALOADER", {})
    cfg.DATALOADER.NUM_WORKERS              = dl_cfg.get("NUM_WORKERS", 4)
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = dl_cfg.get("FILTER_EMPTY_ANNOTATIONS", True)

    solver = yaml_cfg.get("SOLVER", {})
    cfg.SOLVER.BASE_LR           = args.lr       if args.lr       else solver.get("BASE_LR", 0.0001)
    cfg.SOLVER.MAX_ITER          = args.max_iter if args.max_iter else solver.get("MAX_ITER", 10000)
    cfg.SOLVER.STEPS             = tuple(solver.get("STEPS", [7000, 9000]))
    cfg.SOLVER.GAMMA             = solver.get("GAMMA", 0.1)
    cfg.SOLVER.WARMUP_ITERS      = solver.get("WARMUP_ITERS", 500)
    cfg.SOLVER.WARMUP_FACTOR     = solver.get("WARMUP_FACTOR", 0.001)
    cfg.SOLVER.CHECKPOINT_PERIOD = solver.get("CHECKPOINT_PERIOD", 500)
    cfg.SOLVER.IMS_PER_BATCH     = args.batch    if args.batch    else solver.get("IMS_PER_BATCH", 4)
    cfg.SOLVER.WEIGHT_DECAY      = solver.get("WEIGHT_DECAY", 0.0001)
    cfg.SOLVER.MOMENTUM          = solver.get("MOMENTUM", 0.9)
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED    = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
    cfg.SOLVER.AMP.ENABLED       = solver.get("AMP", {}).get("ENABLED", True)

    test_cfg = yaml_cfg.get("TEST", {})
    cfg.TEST.EVAL_PERIOD          = test_cfg.get("EVAL_PERIOD", 500)
    cfg.TEST.DETECTIONS_PER_IMAGE = test_cfg.get("DETECTIONS_PER_IMAGE", 100)

    input_cfg = yaml_cfg.get("INPUT", {})
    cfg.INPUT.MIN_SIZE_TRAIN = (480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800)
    cfg.INPUT.MAX_SIZE_TRAIN = input_cfg.get("MAX_SIZE_TRAIN", 1333)
    cfg.INPUT.MIN_SIZE_TEST  = input_cfg.get("MIN_SIZE_TEST", 800)
    cfg.INPUT.MAX_SIZE_TEST  = input_cfg.get("MAX_SIZE_TEST", 1333)
    cfg.INPUT.RANDOM_FLIP    = input_cfg.get("RANDOM_FLIP", "horizontal")
    cfg.INPUT.MASK_FORMAT    = "bitmask"

    output_dir = args.output_dir if args.output_dir else yaml_cfg.get("OUTPUT_DIR", "outputs/maskrcnn/run")
    cfg.OUTPUT_DIR = output_dir

    if args.resume_from:
        cfg.MODEL.WEIGHTS = args.resume_from

    cfg.SEED = yaml_cfg.get("SEED", 42)
    cfg.freeze()
    return cfg


def find_maskrcnn_checkpoint(output_dir: Path) -> Path:
    final_ckpt = output_dir / "model_final.pth"
    if final_ckpt.exists():
        return final_ckpt
    last_ckpt = output_dir / "last_checkpoint"
    if last_ckpt.exists():
        with open(last_ckpt) as f:
            line = f.readline().strip()
        candidate = output_dir / line
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Aucun checkpoint Mask R-CNN trouvé dans {output_dir}")


def run_maskrcnn_test_eval(model_path: Path, data_root: Path, device: str, output_dir: Path):
    eval_script = Path(__file__).resolve().parent / "evaluate.py"
    cmd = [sys.executable, str(eval_script),
           "--model", str(model_path),
           "--data-root", str(data_root),
           "--model-type", "maskrcnn",
           "--split", "test",
           "--output-dir", str(output_dir)]
    if device is not None:
        cmd += ["--device", str(device)]
    print(f"\n▶ Lancement de l'évaluation test : {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


class FissureTrainer:
    def __init__(self, cfg):
        from detectron2.engine import DefaultTrainer
        from detectron2.evaluation import COCOEvaluator
        from detectron2.data import build_detection_train_loader, DatasetMapper
        from detectron2.data import transforms as T

        class _Trainer(DefaultTrainer):
            @classmethod
            def build_evaluator(cls, cfg, dataset_name, output_folder=None):
                if output_folder is None:
                    output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
                return COCOEvaluator(dataset_name, cfg, True, output_folder)

            @classmethod
            def build_train_loader(cls, cfg):
                # Petit dataset (697 images) : le flip horizontal seul ne suffit pas.
                # On ajoute flip vertical + rotation + luminosité/contraste, adaptés
                # à des fissures qui peuvent apparaître sous n'importe quel angle et
                # avec des conditions d'éclairage variables.
                augmentations = [
                    T.ResizeShortestEdge(
                        cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN,
                        cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
                    ),
                    T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
                    T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
                    T.RandomRotation(angle=[-15, 15], expand=False, sample_style="range"),
                    T.RandomBrightness(0.8, 1.2),
                    T.RandomContrast(0.8, 1.2),
                    T.RandomSaturation(0.9, 1.1),
                ]
                mapper = DatasetMapper(cfg, is_train=True, augmentations=augmentations)
                return build_detection_train_loader(cfg, mapper=mapper)

        self.TrainerCls = _Trainer
        self.cfg = cfg

    def train(self, resume: bool = False):
        from detectron2.utils.logger import setup_logger
        setup_logger()
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        trainer = self.TrainerCls(self.cfg)
        trainer.resume_or_load(resume=resume)
        return trainer.train()


def train(args):
    import_detectron2()

    print("\n" + "="*60)
    print("  Fine-tuning Mask R-CNN — Fissures")
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

    print(f"\n⚙ MaxIter={cfg.SOLVER.MAX_ITER}  LR={cfg.SOLVER.BASE_LR}  "
          f"Batch={cfg.SOLVER.IMS_PER_BATCH}  AMP={cfg.SOLVER.AMP.ENABLED}")
    print(f"   MASK_FORMAT={cfg.INPUT.MASK_FORMAT}  NUM_CLASSES={cfg.MODEL.ROI_HEADS.NUM_CLASSES}")
    print(f"   Output : {cfg.OUTPUT_DIR}")

    resume = args.resume or (args.resume_from is not None)
    print(f"\n🚀 Démarrage (resume={resume})...\n")

    trainer = FissureTrainer(cfg)
    trainer.train(resume=resume)

    print("\n" + "="*60)
    print("  Entraînement terminé")
    print("="*60)
    print(f"  Output : {cfg.OUTPUT_DIR}")

    if args.eval_test:
        try:
            model_path = find_maskrcnn_checkpoint(Path(cfg.OUTPUT_DIR))
            eval_output_dir = Path(cfg.OUTPUT_DIR) / "eval_test"
            run_maskrcnn_test_eval(model_path, data_root, args.device, eval_output_dir)
        except Exception as e:
            print(f"⚠ Échec de l'évaluation test : {e}")

    print(f"\n  Reprise : python scripts/train_maskrcnn.py --data-root <DATA> --resume")


if __name__ == "__main__":
    args = parse_args()
    train(args)
