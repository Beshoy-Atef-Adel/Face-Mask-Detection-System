"""
Face Mask Detection - YOLOv8 Training Script
---------------------------------------------
Trains a YOLOv8n model on the Face Mask Detection dataset
using the local NVIDIA GPU.

Usage:
    python train.py
"""

from ultralytics import YOLO
import torch


def main():
    print("=" * 60)
    print("Face Mask Detection - YOLOv8 Training")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"\nGPU detected: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        device = 0
    else:
        print("\nWARNING: No GPU detected, training on CPU (will be very slow)")
        device = "cpu"

    print("\nStarting training...\n")

    model = YOLO("yolov8s.pt")

    results = model.train(
        data="data.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        device=device,
        name="face_mask_v2",
        patience=30,
        save=True,
        plots=True,
        verbose=True,
        seed=42,
        workers=4,
        cache=False,
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nResults saved to: runs/detect/face_mask_v2/")
    print("Best model: runs/detect/face_mask_v2/weights/best.pt")


if __name__ == "__main__":
    main()
