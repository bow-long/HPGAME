import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "blue_led_dataset"
DATASET_CONFIG = DATASET_DIR / "data.yaml"
MANIFEST_PATH = DATASET_DIR / "review_manifest.json"
BASE_MODEL = "yolo26n.pt"


def main():
    if not DATASET_CONFIG.is_file():
        raise FileNotFoundError(
            f"Dataset config not found: {DATASET_CONFIG}. "
            "Run capture_dataset.py and record some frames first."
        )

    if not MANIFEST_PATH.is_file():
        raise RuntimeError(
            "The dataset has not been manually reviewed. Run review_dataset.py "
            "and review every frame before training."
        )

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    reviewed = set(manifest.get("reviewed", []))
    dataset_images = DATASET_DIR / "images"
    missing_review = []
    for split in ("train", "val"):
        image_dir = dataset_images / split
        images = sorted(image_dir.glob("*.jpg"))
        if not images:
            raise RuntimeError(f"No {split} images found in {image_dir}.")
        for image in images:
            relative_path = image.relative_to(dataset_images).as_posix()
            label = (
                DATASET_DIR
                / "labels"
                / image.relative_to(dataset_images).with_suffix(".txt")
            )
            if relative_path not in reviewed or not label.is_file():
                missing_review.append(relative_path)

    if missing_review:
        raise RuntimeError(
            f"{len(missing_review)} images are not manually reviewed. "
            "Run review_dataset.py to finish reviewing the dataset."
        )

    try:
        from ultralytics import YOLO
        import torch
    except ImportError as error:
        raise RuntimeError(
            "Ultralytics is required. Install project dependencies with "
            "'pip install -r requirements.txt'."
        ) from error

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in this Python environment. Install a "
            "CUDA-enabled PyTorch build and an NVIDIA driver. For CUDA 12.8, "
            "run `pip install --force-reinstall torch==2.11.0+cu128 "
            "--index-url https://download.pytorch.org/whl/cu128`, then verify "
            "torch.cuda.is_available() returns True."
        )

    device_name = torch.cuda.get_device_name(0)
    print(f"Training on NVIDIA GPU: {device_name}")

    try:
        from torchvision.ops import nms

        test_boxes = torch.tensor(
            [[0, 0, 10, 10], [1, 1, 9, 9]],
            dtype=torch.float32,
            device="cuda",
        )
        test_scores = torch.tensor([0.9, 0.8], device="cuda")
        nms(test_boxes, test_scores, 0.5)
    except (ImportError, RuntimeError) as error:
        raise RuntimeError(
            "Torchvision's CUDA NMS operator is unavailable or incompatible "
            "with PyTorch. Reinstall matching CUDA 12.8 builds using the "
            "command in README.md."
        ) from error

    model = YOLO(BASE_MODEL)
    model.train(
        data=str(DATASET_CONFIG),
        epochs=25,
        imgsz=640,
        device=0,
        batch=-1,
        amp=True,
        project=str(BASE_DIR / "runs" / "detect"),
        name="blue_led",
        exist_ok=True,
    )
    print(
        "Training finished. Use runs/detect/blue_led/weights/best.pt "
        "as the tracker model."
    )


if __name__ == "__main__":
    main()
