"""
Script d'entraînement YOLOv11 Segmentation - Fine-tuning Fissures
Exécutable directement sur Google Colab ou en local.
"""

import os
import sys
import shutil
import argparse
import yaml
from pathlib import Path


# ─── Helpers ──────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning YOLOv11 Segmentation pour la détection de fissures")
    parser.add_argument("--data-root", type=str, required=True,
                        help="Chemin vers le dossier du dataset YOLOv11 (contenant train/valid/test), "
                             "ou vers son dossier parent")
    parser.add_argument("--murs-sains", type=str, default=None,
                        help="Chemin vers le dossier d'images de murs sains (exemples négatifs, optionnel)")
    parser.add_argument("--murs-sains-masques", type=str, default=None,
                        help="Chemin vers le dossier de masques noirs des murs sains (optionnel, non requis)")
    parser.add_argument("--config", type=str, default="configs/yolov11_config.yaml",
                        help="Fichier de configuration YAML")
    parser.add_argument("--resume", type=str, default=None,
                        help="Chemin vers un checkpoint pour reprendre l'entraînement (ex: outputs/yolov11/run/weights/last.pt)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Nombre d'époques (surcharge la config)")
    parser.add_argument("--batch", type=int, default=None,
                        help="Taille du batch (surcharge la config)")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="Taille des images (surcharge la config)")
    parser.add_argument("--device", type=str, default=None,
                        help="Appareil : 0, 1, cpu ... (surcharge la config)")
    parser.add_argument("--name", type=str, default=None,
                        help="Nom du run (surcharge la config)")
    return parser.parse_args()


def find_dataset_dir(data_root: Path) -> Path:
    """
    Localise le dossier du dataset YOLOv11 de façon robuste.

    Accepte que --data-root pointe :
      1) directement sur le dossier du dataset (contient train/valid/test) ;
      2) sur un dossier parent contenant un sous-dossier nommé
         'segmentation_fissures.v6i.yolov11' (ancienne convention) ;
      3) sur un dossier parent contenant n'importe quel sous-dossier avec une
         structure train/images + train/labels (détection automatique).
    """
    if not data_root.exists():
        raise FileNotFoundError(
            f"Dossier introuvable : {data_root}\n"
            "Vérifiez le chemin passé à --data-root (le dossier doit être monté/synchronisé depuis Drive)."
        )

    def has_yolo_structure(p: Path) -> bool:
        return (p / "train" / "images").exists() and (p / "train" / "labels").exists()

    # Cas 1 : --data-root pointe déjà sur le dataset
    if has_yolo_structure(data_root):
        return data_root

    # Cas 2 : ancienne convention de nommage exacte
    legacy = data_root / "segmentation_fissures.v6i.yolov11"
    if has_yolo_structure(legacy):
        return legacy

    # Cas 3 : recherche automatique parmi les sous-dossiers directs
    candidates = [d for d in data_root.iterdir() if d.is_dir() and has_yolo_structure(d)]
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
        f"Dataset YOLOv11 introuvable sous : {data_root}\n"
        f"Contenu actuel de ce dossier : {existing}\n"
        "Pointez --data-root directement sur le dossier du dataset Roboflow "
        "(celui qui contient train/, valid/, test/)."
    )


def check_dataset(data_root: Path) -> dict:
    """Vérifie et valide la structure du dataset YOLOv11."""
    dataset_dir = find_dataset_dir(data_root)
    print(f"  Dataset YOLOv11 : {dataset_dir}")

    splits = {}
    for split in ("train", "valid", "test"):
        img_dir = dataset_dir / split / "images"
        lbl_dir = dataset_dir / split / "labels"
        if img_dir.exists() and lbl_dir.exists():
            n_imgs = len(list(img_dir.glob("*.[jp][pn]g")) + list(img_dir.glob("*.jpeg")))
            n_lbls = len(list(lbl_dir.glob("*.txt")))
            splits[split] = {"images": str(img_dir), "labels": str(lbl_dir),
                             "n_images": n_imgs, "n_labels": n_lbls}
            print(f"  [{split:5s}] {n_imgs} images, {n_lbls} labels — {img_dir}")
        else:
            print(f"  [{split:5s}] ABSENT (ignoré)")

    if "train" not in splits:
        raise FileNotFoundError("Le split 'train' est obligatoire.")
    return splits


def add_healthy_walls(healthy_img_dir: Path, splits: dict) -> dict:
    """
    Intègre les murs sains (sans fissures) comme exemples négatifs dans le dataset.

    Pour YOLO, un mur sain se signale simplement par un fichier de label VIDE
    (aucune annotation) : les masques noirs ne sont pas nécessaires et ne sont
    donc pas utilisés ici (voir --murs-sains-masques, conservé uniquement pour
    compatibilité mais ignoré à l'entraînement).
    """
    if healthy_img_dir is None:
        print("  [murs_sains] Non fourni (--murs-sains), étape ignorée.")
        return splits

    if not healthy_img_dir.exists():
        print(f"  [murs_sains] ⚠ Dossier introuvable : {healthy_img_dir} — étape ignorée.")
        return splits

    img_exts = {".jpg", ".jpeg", ".png"}
    # Le dossier peut être plat (images directement dedans) ou contenir des
    # sous-dossiers train/valid/test : dans les deux cas on ne prend que les
    # images pour le split train.
    if (healthy_img_dir / "train").exists():
        search_dir = healthy_img_dir / "train"
    else:
        search_dir = healthy_img_dir
    images = [p for p in search_dir.iterdir() if p.is_file() and p.suffix.lower() in img_exts]

    if not images:
        print(f"  [murs_sains] ⚠ Aucune image trouvée dans {search_dir} — étape ignorée.")
        return splits

    # Copie dans le split train
    train_img_dest = Path(splits["train"]["images"])
    train_lbl_dest = Path(splits["train"]["labels"])

    copied = 0
    for img_path in images:
        dest_img = train_img_dest / img_path.name
        dest_lbl = train_lbl_dest / f"{img_path.stem}.txt"
        if not dest_img.exists():
            shutil.copy2(img_path, dest_img)
        # Mur sain = pas d'annotation → fichier label vide (classe absente)
        if not dest_lbl.exists():
            dest_lbl.write_text("")
        copied += 1

    print(f"  [murs_sains] ✓ {copied} image(s) intégrée(s) dans le split train")
    splits["train"]["n_images"] += copied
    return splits


def build_dataset_yaml(splits: dict, output_path: Path) -> str:
    """Génère le fichier data YAML requis par Ultralytics."""
    data = {
        "path": str(Path(splits["train"]["images"]).parent.parent),
        "train": "train/images",
        "val":   splits.get("valid", splits["train"])["images"].replace(
                     str(Path(splits["train"]["images"]).parent.parent) + "/", ""),
        "nc": 1,
        "names": ["fissure"],
    }
    if "test" in splits:
        data["test"] = splits["test"]["images"].replace(
            str(Path(splits["train"]["images"]).parent.parent) + "/", "")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    print(f"  Dataset YAML généré : {output_path}")
    return str(output_path)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ─── Entraînement ─────────────────────────────────────────────────────────────

def train(args):
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("❌ ultralytics non installé. Exécutez : pip install ultralytics")

    print("\n" + "="*60)
    print("  Fine-tuning YOLOv11 Segmentation — Détection de fissures")
    print("="*60 + "\n")

    # Chargement de la config
    cfg = load_config(args.config)

    # Surcharges CLI
    overrides = {}
    if args.epochs  is not None: overrides["epochs"]  = args.epochs
    if args.batch   is not None: overrides["batch"]   = args.batch
    if args.imgsz   is not None: overrides["imgsz"]   = args.imgsz
    if args.device  is not None: overrides["device"]  = args.device
    if args.name    is not None: overrides["name"]    = args.name

    data_root = Path(args.data_root)
    print("📂 Vérification du dataset...")
    splits = check_dataset(data_root)

    print("🌿 Intégration des murs sains...")
    healthy_dir = Path(args.murs_sains) if args.murs_sains else None
    splits = add_healthy_walls(healthy_dir, splits)

    print("📝 Génération du fichier dataset YAML...")
    dataset_yaml = build_dataset_yaml(splits, Path("configs/dataset_yolo.yaml"))

    # Modèle : reprise ou démarrage frais
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            sys.exit(f"❌ Checkpoint introuvable : {resume_path}")
        print(f"\n▶ Reprise de l'entraînement depuis : {resume_path}")
        model = YOLO(str(resume_path))
        train_kwargs = dict(resume=True)
    else:
        model_name = cfg.get("model", "yolo11x-seg.pt")
        print(f"\n▶ Chargement du modèle de base : {model_name}")
        model = YOLO(model_name)
        train_kwargs = {}

    # Construction des arguments d'entraînement
    train_kwargs.update({
        "data":             dataset_yaml,
        "epochs":           overrides.get("epochs",  cfg.get("epochs", 100)),
        "patience":         cfg.get("patience", 20),
        "batch":            overrides.get("batch",   cfg.get("batch", 8)),
        "imgsz":            overrides.get("imgsz",   cfg.get("imgsz", 640)),
        "optimizer":        cfg.get("optimizer", "AdamW"),
        "lr0":              cfg.get("lr0", 0.0005),
        "lrf":              cfg.get("lrf", 0.01),
        "momentum":         cfg.get("momentum", 0.937),
        "weight_decay":     cfg.get("weight_decay", 0.0005),
        "warmup_epochs":    cfg.get("warmup_epochs", 3),
        "warmup_momentum":  cfg.get("warmup_momentum", 0.8),
        "warmup_bias_lr":   cfg.get("warmup_bias_lr", 0.1),
        "box":              cfg.get("box", 7.5),
        "cls":              cfg.get("cls", 0.5),
        "dfl":              cfg.get("dfl", 1.5),
        "label_smoothing":  cfg.get("label_smoothing", 0.0),
        "nbs":              cfg.get("nbs", 64),
        "overlap_mask":     cfg.get("overlap_mask", True),
        "mask_ratio":       cfg.get("mask_ratio", 4),
        "dropout":          cfg.get("dropout", 0.1),
        "amp":              cfg.get("amp", True),
        "hsv_h":            cfg.get("hsv_h", 0.015),
        "hsv_s":            cfg.get("hsv_s", 0.7),
        "hsv_v":            cfg.get("hsv_v", 0.4),
        "degrees":          cfg.get("degrees", 10.0),
        "translate":        cfg.get("translate", 0.1),
        "scale":            cfg.get("scale", 0.5),
        "shear":            cfg.get("shear", 2.0),
        "perspective":      cfg.get("perspective", 0.0001),
        "flipud":           cfg.get("flipud", 0.3),
        "fliplr":           cfg.get("fliplr", 0.5),
        "mosaic":           cfg.get("mosaic", 1.0),
        "mixup":            cfg.get("mixup", 0.1),
        "copy_paste":       cfg.get("copy_paste", 0.1),
        "erasing":          cfg.get("erasing", 0.4),
        "save_period":      cfg.get("save_period", 10),
        "save":             True,
        "save_json":        True,
        "plots":            True,
        "val":              True,
        "device":           overrides.get("device", cfg.get("device", 0)),
        "workers":          cfg.get("workers", 4),
        "project":          cfg.get("project", "outputs/yolov11"),
        "name":             overrides.get("name", cfg.get("name", "run")),
        "exist_ok":         cfg.get("exist_ok", False),
        "verbose":          True,
        "seed":             cfg.get("seed", 42),
        "deterministic":    cfg.get("deterministic", True),
        "single_cls":       cfg.get("single_cls", True),
        "retina_masks":     cfg.get("retina_masks", True),
        "conf":             cfg.get("conf", 0.25),
        "iou":              cfg.get("iou", 0.45),
    })

    print(f"\n⚙ Hyperparamètres principaux :")
    print(f"   Epochs     : {train_kwargs['epochs']}")
    print(f"   Patience   : {train_kwargs['patience']}")
    print(f"   Batch      : {train_kwargs['batch']}")
    print(f"   Image size : {train_kwargs['imgsz']}")
    print(f"   LR0        : {train_kwargs['lr0']}")
    print(f"   Optimizer  : {train_kwargs['optimizer']}")
    print(f"   AMP        : {train_kwargs['amp']}")
    print(f"   Device     : {train_kwargs['device']}")

    print("\n🚀 Démarrage de l'entraînement...\n")
    results = model.train(**train_kwargs)

    # Résumé post-entraînement
    print("\n" + "="*60)
    print("  Entraînement terminé")
    print("="*60)
    project_dir = Path(train_kwargs["project"]) / train_kwargs["name"]
    best_weights = project_dir / "weights" / "best.pt"
    last_weights = project_dir / "weights" / "last.pt"
    print(f"  Meilleurs poids : {best_weights}")
    print(f"  Derniers poids  : {last_weights}")
    print(f"  Résultats       : {project_dir}")
    print("\n  Pour reprendre l'entraînement :")
    print(f"  python scripts/train_yolov11.py --data-root <DATA> --resume {last_weights}")

    return results


# ─── Point d'entrée ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    train(args)
