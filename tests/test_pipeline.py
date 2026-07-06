import json
from pathlib import Path

import yaml

from scripts.evaluate import detect_model_type, normalize_maskrcnn_metrics
from scripts.train_maskrcnn import check_coco_dataset
from scripts.train_yolov11 import build_dataset_yaml, check_dataset


def _make_yolo_split(root: Path, split: str) -> None:
    split_dir = root / split
    (split_dir / "images").mkdir(parents=True, exist_ok=True)
    (split_dir / "labels").mkdir(parents=True, exist_ok=True)
    (split_dir / "images" / f"{split}_img.jpg").write_bytes(b"fake-image")
    (split_dir / "labels" / f"{split}_img.txt").write_text("0 0.1 0.2 0.3 0.4\n")


def _make_coco_split(root: Path, split: str) -> None:
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    ann_path = split_dir / "_annotations.coco.json"
    coco = {
        "images": [{"id": 1, "file_name": f"{split}_img.jpg", "height": 100, "width": 100}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0}],
        "categories": [{"id": 1, "name": "fissure"}],
    }
    ann_path.write_text(json.dumps(coco))


def test_build_dataset_yaml_uses_valid_split_and_includes_test(tmp_path):
    data_root = tmp_path / "yolo_dataset"
    data_root.mkdir(parents=True)
    _make_yolo_split(data_root, "train")
    _make_yolo_split(data_root, "valid")
    _make_yolo_split(data_root, "test")

    splits = check_dataset(data_root)
    output_path = tmp_path / "dataset.yaml"
    build_dataset_yaml(splits, output_path)

    data = yaml.safe_load(output_path.read_text())
    assert data["val"] == "valid/images"
    assert data["test"] == "test/images"
    assert data["path"] == str(data_root)


def test_check_coco_dataset_detects_train_valid_test_splits(tmp_path):
    data_root = tmp_path / "coco_dataset"
    data_root.mkdir(parents=True)
    _make_coco_split(data_root, "train")
    _make_coco_split(data_root, "valid")
    _make_coco_split(data_root, "test")

    splits, dataset_dir = check_coco_dataset(data_root)

    assert dataset_dir == data_root
    assert set(splits) == {"train", "valid", "test"}
    assert splits["train"]["n_images"] == 1
    assert splits["valid"]["n_annotations"] == 1
    assert splits["test"]["n_images"] == 1


def test_normalize_maskrcnn_metrics_converts_coco_results_and_f1():
    raw_results = {
        "bbox": {"AP50": 80.0, "AP": 60.0, "AP75": 40.0, "AR@100": 70.0},
        "segm": {"AP50": 75.5, "AP": 58.0, "AP75": 35.0, "AR@100": 66.0},
    }

    metrics = normalize_maskrcnn_metrics(raw_results)

    assert metrics["box_map50"] == 0.8
    assert metrics["box_map5095"] == 0.6
    assert metrics["box_precision"] == 0.8
    assert metrics["box_recall"] == 0.7
    assert metrics["box_f1"] == 0.7467
    assert metrics["mask_map50"] == 0.755
    assert metrics["mask_recall"] == 0.66
    assert metrics["mask_f1"] == 0.7043


def test_detect_model_type_handles_pt_and_pth_files():
    assert detect_model_type("weights.pt") == "yolo"
    assert detect_model_type("weights.pth") == "maskrcnn"


def test_crack_configs_use_smaller_targets_for_thin_cracks():
    yolov11_cfg = yaml.safe_load(Path("configs/yolov11_config.yaml").read_text())
    maskrcnn_cfg = yaml.safe_load(Path("configs/maskrcnn_config.yaml").read_text())

    assert yolov11_cfg["model"] == "yolo11m-seg.pt"
    assert maskrcnn_cfg["MODEL"]["ANCHOR_GENERATOR"]["SIZES"][0][0] <= 8
