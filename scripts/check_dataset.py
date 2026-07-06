"""
Vérification rapide d'un dataset (YOLOv11 ou COCO) avant de lancer un entraînement complet.

Usage :
    python scripts/check_dataset.py --data-root /content/drive/MyDrive/segmentation_fissures.v6i.yolov11
    python scripts/check_dataset.py --data-root /content/drive/MyDrive/segmentation_fissures.v7i.coco-segmentation
    python scripts/check_dataset.py --data-root /content/drive/MyDrive/mon_dossier --murs-sains /content/drive/MyDrive/murs_sains

Le format (YOLOv11 ou COCO) est détecté automatiquement.
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--murs-sains", type=str, default=None)
    return parser.parse_args()


def check_yolo(data_root: Path):
    import train_yolov11 as t
    print("\n▶ Format détecté : YOLOv11 (segmentation)")
    splits = t.check_dataset(data_root)
    return splits


def check_coco(data_root: Path):
    import train_maskrcnn as t
    print("\n▶ Format détecté : COCO (segmentation)")
    splits, dataset_dir = t.check_coco_dataset(data_root)
    return splits


def check_healthy_walls(murs_sains: str):
    if not murs_sains:
        return
    healthy_dir = Path(murs_sains)
    print(f"\n🌿 Murs sains : {healthy_dir}")
    if not healthy_dir.exists():
        print(f"  ⚠ Dossier introuvable : {healthy_dir}")
        return
    img_exts = {".jpg", ".jpeg", ".png"}
    search_dir = healthy_dir / "train" if (healthy_dir / "train").exists() else healthy_dir
    images = [p for p in search_dir.iterdir() if p.is_file() and p.suffix.lower() in img_exts]
    print(f"  ✓ {len(images)} image(s) trouvée(s) dans {search_dir}")


def main():
    args = parse_args()
    data_root = Path(args.data_root)

    print("=" * 60)
    print("  Vérification du dataset")
    print("=" * 60)

    is_yolo = (data_root / "train" / "labels").exists() or any(
        (d / "train" / "labels").exists() for d in data_root.iterdir() if data_root.exists() and d.is_dir()
    ) if data_root.exists() else False

    is_coco = (data_root / "train" / "_annotations.coco.json").exists() or any(
        (d / "train" / "_annotations.coco.json").exists() for d in data_root.iterdir() if data_root.exists() and d.is_dir()
    ) if data_root.exists() else False

    try:
        if is_coco and not is_yolo:
            check_coco(data_root)
        elif is_yolo and not is_coco:
            check_yolo(data_root)
        else:
            print("\n⚠ Format ambigu ou dataset introuvable. Tentative YOLOv11 puis COCO...")
            try:
                check_yolo(data_root)
            except FileNotFoundError:
                check_coco(data_root)
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        sys.exit(1)

    check_healthy_walls(args.murs_sains)

    print("\n" + "=" * 60)
    print("  ✓ Dataset valide — prêt pour l'entraînement")
    print("=" * 60)


if __name__ == "__main__":
    main()
