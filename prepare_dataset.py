"""
Face Mask Detection - Dataset Preparation Script
-------------------------------------------------
Converts Pascal VOC XML annotations to YOLO TXT format,
and splits the dataset into train/val (80/20).

Usage:
    python prepare_dataset.py
"""

import os
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================
PROJECT_ROOT = Path(__file__).parent
IMAGES_SRC = PROJECT_ROOT / "data" / "images"
ANNOT_SRC = PROJECT_ROOT / "data" / "annotations"

DATASET_OUT = PROJECT_ROOT / "dataset"
TRAIN_RATIO = 0.8
SEED = 42

CLASSES = ["with_mask", "without_mask", "mask_weared_incorrect"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASSES)}


def convert_bbox_to_yolo(size, box):
    """Convert Pascal VOC bbox to YOLO format (normalized cx, cy, w, h)."""
    img_w, img_h = size
    xmin, ymin, xmax, ymax = box

    cx = (xmin + xmax) / 2.0 / img_w
    cy = (ymin + ymax) / 2.0 / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    return cx, cy, w, h


def parse_xml(xml_path):
    """Parse a Pascal VOC XML file and return image size and YOLO-format lines."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    yolo_lines = []
    for obj in root.findall("object"):
        cls_name = obj.find("name").text.strip()
        if cls_name not in CLASS_TO_ID:
            print(f"  WARNING: unknown class '{cls_name}' in {xml_path.name}, skipping")
            continue

        cls_id = CLASS_TO_ID[cls_name]
        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        cx, cy, w, h = convert_bbox_to_yolo((img_w, img_h), (xmin, ymin, xmax, ymax))
        yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return yolo_lines


def setup_output_dirs():
    """Create the YOLO-style output directory structure."""
    if DATASET_OUT.exists():
        print(f"Removing existing {DATASET_OUT}")
        shutil.rmtree(DATASET_OUT)

    for split in ["train", "val"]:
        (DATASET_OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_OUT / "labels" / split).mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 60)
    print("Face Mask Detection - Dataset Preparation")
    print("=" * 60)

    if not IMAGES_SRC.exists() or not ANNOT_SRC.exists():
        print(f"ERROR: source folders not found")
        print(f"  images: {IMAGES_SRC}")
        print(f"  annotations: {ANNOT_SRC}")
        return

    setup_output_dirs()

    image_files = sorted([f for f in IMAGES_SRC.iterdir() if f.suffix.lower() in [".png", ".jpg", ".jpeg"]])
    print(f"\nFound {len(image_files)} images")

    matched = []
    for img_path in image_files:
        xml_path = ANNOT_SRC / f"{img_path.stem}.xml"
        if xml_path.exists():
            matched.append((img_path, xml_path))

    print(f"Matched {len(matched)} image/annotation pairs")

    random.seed(SEED)
    random.shuffle(matched)
    n_train = int(len(matched) * TRAIN_RATIO)
    train_set = matched[:n_train]
    val_set = matched[n_train:]

    print(f"\nSplit: {len(train_set)} train / {len(val_set)} val")
    print("\nConverting and copying files...")

    class_counts = {c: 0 for c in CLASSES}

    for split_name, split_data in [("train", train_set), ("val", val_set)]:
        for img_path, xml_path in split_data:
            yolo_lines = parse_xml(xml_path)

            for line in yolo_lines:
                cls_id = int(line.split()[0])
                class_counts[CLASSES[cls_id]] += 1

            shutil.copy2(img_path, DATASET_OUT / "images" / split_name / img_path.name)
            label_path = DATASET_OUT / "labels" / split_name / f"{img_path.stem}.txt"
            with open(label_path, "w") as f:
                f.write("\n".join(yolo_lines))

    data_yaml_path = DATASET_OUT / "data.yaml"
    with open(data_yaml_path, "w") as f:
        f.write(f"path: {DATASET_OUT.as_posix()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")
        f.write(f"nc: {len(CLASSES)}\n")
        f.write(f"names: {CLASSES}\n")

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"\nOutput: {DATASET_OUT}")
    print(f"\nClass distribution (total objects):")
    for cls, count in class_counts.items():
        print(f"  {cls}: {count}")
    print(f"\ndata.yaml created at: {data_yaml_path}")
    print("\nNext step: zip the 'dataset' folder and upload to Google Colab")


if __name__ == "__main__":
    main()
