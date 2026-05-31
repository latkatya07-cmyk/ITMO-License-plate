import os
import sys
import yaml
import random
import argparse
import logging
from pathlib import Path

import cv2
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(r"C:\Users\elatypova002\Desktop\ИТМО\2 семестр 1 курс\Проектирование архитектур нейронных систем\ДЗ 1")
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_PATH = PROJECT_ROOT / "license_plate_config.yaml"
RESULTS_DIR = PROJECT_ROOT / "results"


def setup_environment():
    for subdir in ["predictions", "evaluation", "external_predictions", "training"]:
        (RESULTS_DIR / subdir).mkdir(parents=True, exist_ok=True)


def load_config(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def train(model_name: str, epochs: int, imgsz: int, batch: int, device: str) -> YOLO:
    logger.info(f"Training {model_name} for {epochs} epochs")
    
    model = YOLO(model_name)
    model.train(
        data=str(CONFIG_PATH),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(RESULTS_DIR / "training"),
        name="license_plate_detection",
        save=True,
        save_period=10,
        verbose=False
    )
    return model


def evaluate(model: YOLO) -> dict:
    logger.info("Running evaluation on test set")
    
    metrics = model.val(
        data=str(CONFIG_PATH),
        split='test',
        save_json=True,
        project=str(RESULTS_DIR / "evaluation"),
        name="test_metrics",
        verbose=False
    )
    
    results = {
        'mAP@0.5': float(metrics.box.map50),
        'mAP@0.5:0.95': float(metrics.box.map),
        'Precision': float(metrics.box.mp),
        'Recall': float(metrics.box.mr),
        'F1-score': float(metrics.box.f1)
    }
    
    output_path = RESULTS_DIR / "evaluation" / "metrics.yaml"
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(results, f, allow_unicode=True)
    
    logger.info(f"Metrics saved to {output_path}")
    return results


def analyze_predictions(model: YOLO, sample_size: int = 20) -> list:
    logger.info("Analyzing prediction errors")
    
    test_dir = DATA_DIR / "images" / "test"
    images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
    
    if not images:
        logger.warning("No test images found")
        return []
    
    sample = random.sample(images, min(sample_size, len(images)))
    errors = []
    
    for img_path in sample:
        results = model.predict(str(img_path), conf=0.25, iou=0.45, verbose=False)
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        annotated = results[0].plot()
        output = RESULTS_DIR / "predictions" / f"pred_{img_path.name}"
        cv2.imwrite(str(output), annotated)
        
        boxes = results[0].boxes
        if len(boxes) == 0:
            errors.append((img_path.name, "no_detection"))
        elif any(box.conf[0] < 0.5 for box in boxes):
            errors.append((img_path.name, "low_confidence"))
        elif len(boxes) > 1:
            errors.append((img_path.name, "multiple_detections"))
    
    if errors:
        pd.DataFrame(errors, columns=['image', 'error_type']).to_csv(
            RESULTS_DIR / "error_analysis.csv", index=False
        )
        logger.info(f"Error report saved: {len(errors)} cases flagged")
    
    return errors


def test_external(model: YOLO, external_dir: Path = None):
    external_dir = external_dir or (PROJECT_ROOT / "external_test")
    external_dir.mkdir(exist_ok=True)
    
    images = list(external_dir.glob("*.jpg")) + list(external_dir.glob("*.png"))
    if not images:
        logger.info(f"Add images to {external_dir} for external testing")
        return
    
    output_dir = RESULTS_DIR / "external_predictions"
    output_dir.mkdir(exist_ok=True)
    
    for img_path in images:
        results = model.predict(str(img_path), conf=0.25, iou=0.45, verbose=False)
        annotated = results[0].plot()
        output = output_dir / f"ext_{img_path.name}"
        cv2.imwrite(str(output), annotated)
        logger.info(f"Processed: {img_path.name}")


def plot_curves():
    results_csv = RESULTS_DIR / "training" / "license_plate_detection-2" / "results.csv"
    if not results_csv.exists():
        logger.warning(f"results.csv not found: {results_csv}")
        return
    
    df = pd.read_csv(results_csv)
    logger.info(f"Loaded {len(df)} epochs from {results_csv}")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Metrics', fontsize=16, fontweight='bold')
    
    axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Box')
    axes[0, 0].plot(df['epoch'], df['train/cls_loss'], label='Cls')
    axes[0, 0].plot(df['epoch'], df['train/dfl_loss'], label='DFL')
    axes[0, 0].set_title('Training Losses')
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(df['epoch'], df['metrics/precision(B)'], label='Precision')
    axes[0, 1].plot(df['epoch'], df['metrics/recall(B)'], label='Recall')
    axes[0, 1].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5')
    axes[0, 1].set_title('Validation Metrics')
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(df['epoch'], df['metrics/mAP50-95(B)'], color='purple', linewidth=2)
    axes[1, 0].set_title('mAP@0.5:0.95')
    axes[1, 0].grid(True, alpha=0.3)
    
    p = df['metrics/precision(B)'].values
    r = df['metrics/recall(B)'].values
    f1 = 2 * p * r / (p + r + 1e-8)
    axes[1, 1].plot(df['epoch'], f1, color='orange', linewidth=2)
    axes[1, 1].set_title('F1-score (calculated)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = RESULTS_DIR / "training_curves.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def get_model_path() -> Path:
    return RESULTS_DIR / "training" / "license_plate_detection" / "weights" / "best.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--analyze', action='store_true')
    parser.add_argument('--test-external', action='store_true')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--model', default='yolov8n.pt')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--device', default='0')
    args = parser.parse_args()
    
    setup_environment()
    
    if args.train or args.all:
        model = train(args.model, args.epochs, 640, 16, args.device)
        plot_curves()
    else:
        model_path = get_model_path()
        if not model_path.exists():
            logger.error("Model not found. Run with --train first.")
            return
        model = YOLO(str(model_path))
    
    if args.eval or args.all:
        evaluate(model)
    
    if args.analyze or args.all:
        analyze_predictions(model)
    
    if args.test_external or args.all:
        test_external(model)
    
    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()