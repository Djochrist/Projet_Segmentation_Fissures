import os
import sys
import shutil
import argparse
import subprocess
import yaml
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--murs-sains", type=str, default=None)
    parser.add_argument("--murs-sains-masques", type=str, default=None)
    parser.add_argument("--config", type=str, default="configs/yolov11_config.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--eval-test", action="store_true",
                        help="Évaluer automatiquement le split test après l'entraînement")
    return parser.parse_args()


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


def find_dataset_dir(data_root: Path) -> Path:
    if not data_root.exists():
        raise FileNotFoundError(
            f"Dossier introuvable : {data_root}\n"
            f"Vérifie le chemin exact de ton dataset sur Drive :\n{_drive_hint(data_root)}"
        )

    def has_yolo_structure(p: Path) -> bool:
        return (p / "train" / "images").exists() and (p / "train" / "labels").exists()

    if has_yolo_structure(data_root):
        return data_root

    legacy = data_root / "segmentation_fissures.v6i.yolov11"
    if has_yolo_structure(legacy):
        return legacy

    candidates = [d for d in data_root.iterdir() if d.is_dir() and has_yolo_structure(d)]
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
        f"Dataset YOLOv11 introuvable sous : {data_root}\nContenu : {existing}\n"
        f"{_drive_hint(data_root)}"
    )


def check_dataset(data_root: Path) -> dict:
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
            print(f"  [{split:5s}] {n_imgs} images, {n_lbls} labels")
        else:
            print(f"  [{split:5s}] ABSENT (ignoré)")
    if "train" not in splits:
        raise FileNotFoundError("Le split 'train' est obligatoire.")
    return splits


def add_healthy_walls(healthy_img_dir: Path, splits: dict) -> dict:
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

    train_img_dest = Path(splits["train"]["images"])
    train_lbl_dest = Path(splits["train"]["labels"])
    copied = 0
    for img_path in images:
        dest_img = train_img_dest / img_path.name
        dest_lbl = train_lbl_dest / f"{img_path.stem}.txt"
        if not dest_img.exists():
            shutil.copy2(img_path, dest_img)
        if not dest_lbl.exists():
            dest_lbl.write_text("")
        copied += 1

    print(f"  [murs_sains] ✓ {copied} image(s) intégrée(s)")
    splits["train"]["n_images"] += copied
    return splits


def build_dataset_yaml(splits: dict, output_path: Path) -> str:
    data = {
        "path": str(Path(splits["train"]["images"]).parent.parent),
        "train": "train/images",
        "val": splits.get("valid", splits["train"])["images"].replace(
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
    print(f"  Dataset YAML : {output_path}")
    return str(output_path)


def find_yolo_checkpoint(project_dir: Path) -> Path:
    best = project_dir / "weights" / "best.pt"
    last = project_dir / "weights" / "last.pt"
    if best.exists():
        return best
    if last.exists():
        return last
    raise FileNotFoundError(f"Aucun checkpoint YOLO trouvé dans {project_dir / 'weights'}")


def run_yolo_test_eval(model_path: Path, data_root: Path, device: str, output_dir: Path):
    eval_script = Path(__file__).resolve().parent / "evaluate.py"
    cmd = [sys.executable, str(eval_script),
           "--model", str(model_path),
           "--data-root", str(data_root),
           "--model-type", "yolo",
           "--split", "test",
           "--output-dir", str(output_dir)]
    if device is not None:
        cmd += ["--device", str(device)]
    print(f"\n▶ Lancement de l'évaluation test : {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def train(args):
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("❌ ultralytics non installé : pip install ultralytics")

    print("\n" + "="*60)
    print("  Fine-tuning YOLOv11 Segmentation — Fissures")
    print("="*60 + "\n")

    cfg = load_config(args.config)

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

    print("📝 Génération du dataset YAML...")
    dataset_yaml = build_dataset_yaml(splits, Path("configs/dataset_yolo.yaml"))

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            sys.exit(f"❌ Checkpoint introuvable : {resume_path}")
        print(f"\n▶ Reprise depuis : {resume_path}")
        model = YOLO(str(resume_path))
        train_kwargs = dict(resume=True)
    else:
        model_name = cfg.get("model", "yolo11m-seg.pt")
        print(f"\n▶ Modèle de base : {model_name}")
        model = YOLO(model_name)
        train_kwargs = {}

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

    print(f"\n⚙ Epochs={train_kwargs['epochs']}  Batch={train_kwargs['batch']}  "
          f"Imgsz={train_kwargs['imgsz']}  LR={train_kwargs['lr0']}  "
          f"AMP={train_kwargs['amp']}")
    print("\n🚀 Démarrage de l'entraînement...\n")
    results = model.train(**train_kwargs)

    print("\n" + "="*60)
    print("  Entraînement terminé")
    print("="*60)
    project_dir = Path(train_kwargs["project"]) / train_kwargs["name"]
    print(f"  Meilleurs poids : {project_dir / 'weights' / 'best.pt'}")
    print(f"  Derniers poids  : {project_dir / 'weights' / 'last.pt'}")

    if args.eval_test:
        try:
            model_path = find_yolo_checkpoint(project_dir)
            eval_output_dir = project_dir / "eval_test"
            run_yolo_test_eval(model_path, data_root, overrides.get("device", cfg.get("device", 0)), eval_output_dir)
        except Exception as e:
            print(f"⚠ Échec de l'évaluation test : {e}")

    print(f"\n  Reprise : python scripts/train_yolov11.py --data-root <DATA> "
          f"--resume {project_dir / 'weights' / 'last.pt'}")
    return results


if __name__ == "__main__":
    args = parse_args()
    train(args)
