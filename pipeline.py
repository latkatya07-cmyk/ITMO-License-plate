import sys
import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
from train_crnn import CRNN, load_alphabet, decode_sequence

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_H, IMG_W = 32, 100

def run_pipeline(image_path: str, conf_thresh: float = 0.4) -> list:
    det_path = Path("results/training/license_plate_detection-2/weights/best.pt")
    rec_path = Path("autoriaNumberplateOcrRu/mini_dataset/crnn_baseline.pt")
    alpha_path = Path("autoriaNumberplateOcrRu/mini_dataset/alphabet.txt")

    detector = YOLO(str(det_path))
    char_to_idx, idx_to_char = load_alphabet(alpha_path)
    blank_idx = char_to_idx["<blank>"]
    num_classes = len(char_to_idx)

    recognizer = CRNN(IMG_H, 1, num_classes).to(DEVICE)
    recognizer.load_state_dict(torch.load(str(rec_path), map_location=DEVICE, weights_only=True))
    recognizer.eval()

    img = cv2.imread(str(image_path))

    results = detector(img, conf=conf_thresh, verbose=False)
    detections = []

    for res in results:
        boxes = res.boxes
        if boxes is None:
            continue
        for box, conf in zip(boxes.xyxy.cpu(), boxes.conf.cpu()):
            if conf < conf_thresh:
                continue
            x1, y1, x2, y2 = map(int, box)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).convert("L")
            crop_pil = crop_pil.resize((IMG_W, IMG_H), Image.BICUBIC)
            inp = np.array(crop_pil, dtype=np.float32) / 255.0
            inp = torch.from_numpy(inp).unsqueeze(0).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                out = recognizer(inp)
                pred_ids = out.argmax(dim=2).squeeze(1)
                text = decode_sequence(pred_ids, idx_to_char, blank_idx)

            detections.append({"text": text, "conf": float(conf), "bbox": [x1, y1, x2, y2]})
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{text} ({conf:.2f})", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    out_path = Path("pipeline_result.jpg")
    cv2.imwrite(str(out_path), img)
    
    return detections

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <car_image.jpg>")
        sys.exit(1)

    img_path = sys.argv[1]
    print(f"🔍 Running end-to-end pipeline on: {img_path}")
    results = run_pipeline(img_path)

    if not results:
        print("Номерные знаки не обнаружены.")
    else:
        for det in results:
            print(f"'{det['text']}' (conf: {det['conf']:.2f}, bbox: {det['bbox']})")