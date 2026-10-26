import os
import shutil
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
try:
    from src.utils import get_file_hash
except ImportError:
    from utils import get_file_hash

def auto_crop_strip(img_path, model, conf_thresh=0.20, sample_collector=None):
    """
    Stage 1: Localizes and crops nitrocellulose strips from raw assay images
    using fine-tuned YOLOv8. Preserves native color profiles and EXIF orientation.
    """
    raw_bgr = cv2.imread(img_path)
    if raw_bgr is None:
        raw_bgr = np.array(Image.open(img_path).convert('RGB'))[:, :, ::-1]

    img_h, img_w = raw_bgr.shape[:2]
    results = model.predict(source=raw_bgr, conf=conf_thresh, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy() if len(results) > 0 and len(results[0].boxes) > 0 else []

    if len(boxes) > 0:
        best_box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        x1, y1, x2, y2 = map(int, best_box[:4])

        if sample_collector is not None:
            sample_collector.append((img_path, x1, y1, x2, y2, img_w, img_h))

        x_start = max(0, x1)
        x_end = min(img_w, x2)
        y_start = max(0, y1)
        y_end = min(img_h, y2)

        cropped_bgr = raw_bgr[y_start:y_end, x_start:x_end]
        return cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)

    return cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)

def detect_bands_stage2(cropped_rgb, band_model, conf_thresh=0.25):
    """
    Stage 2: Detects Control (C, class 0) and Test (T, class 1) bands on a cropped strip.
    Returns best candidate dictionaries for C and T lines with bounding box coordinates,
    centroid y-position (yc), and detection confidence.
    """
    cropped_bgr = cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2BGR)
    results = band_model.predict(source=cropped_bgr, conf=conf_thresh, verbose=False)

    c_candidates = []
    t_candidates = []

    if len(results) > 0 and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            cls_id = int(box.cls[0].cpu().item())
            conf = float(box.conf[0].cpu().item())
            xyxy = box.xyxy[0].cpu().numpy()
            det = {
                'box': xyxy,
                'conf': conf,
                'yc': float((xyxy[1] + xyxy[3]) / 2.0),
                'y1': float(xyxy[1]),
                'y2': float(xyxy[3]),
                'x1': float(xyxy[0]),
                'x2': float(xyxy[2])
            }
            if cls_id == 0:
                c_candidates.append(det)
            elif cls_id == 1:
                t_candidates.append(det)

    best_c = max(c_candidates, key=lambda d: d['conf']) if c_candidates else None
    best_t = max(t_candidates, key=lambda d: d['conf']) if t_candidates else None

    # Spatial sanity check: Control line must appear upstream of Test line (smaller Y)
    if best_c and best_t and best_c['yc'] > best_t['yc']:
        best_c, best_t = best_t, best_c

    return best_c, best_t

def fine_tune_model(dataset_dir, weights_path, collected_samples):
    """
    Fine-tunes the YOLO model with new unique samples using MD5 checksum validation.
    """
    if not collected_samples:
        print("\nNo newly cropped strips to learn from.")
        return

    yaml_file = os.path.join(dataset_dir, "data.yaml")
    if not os.path.exists(yaml_file):
        print(f"\n[Warning] Dataset config not found at '{yaml_file}'. Skipping auto-retraining.")
        return

    train_img_dir = os.path.join(dataset_dir, "images", "train")
    train_lbl_dir = os.path.join(dataset_dir, "labels", "train")
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_lbl_dir, exist_ok=True)

    registry_file = os.path.join(dataset_dir, "processed_hashes.txt")
    existing_hashes = set()
    if os.path.exists(registry_file):
        with open(registry_file, "r") as f:
            existing_hashes = set(line.strip() for line in f if line.strip())

    new_samples_added = 0
    new_hashes_to_record = []

    for img_p, x1, y1, x2, y2, img_w, img_h in collected_samples:
        img_hash = get_file_hash(img_p)
        if img_hash in existing_hashes:
            continue

        base_name = os.path.splitext(os.path.basename(img_p))[0]
        ext = os.path.splitext(img_p)[1]
        unique_name = f"auto_{base_name}_{img_hash[:8]}"

        dest_img = os.path.join(train_img_dir, f"{unique_name}{ext}")
        shutil.copy(img_p, dest_img)

        # Convert to YOLO normalized format (x_center, y_center, width, height)
        xc = ((x1 + x2) / 2.0) / img_w
        yc = ((y1 + y2) / 2.0) / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h

        dest_lbl = os.path.join(train_lbl_dir, f"{unique_name}.txt")
        with open(dest_lbl, "w") as f:
            f.write(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        existing_hashes.add(img_hash)
        new_hashes_to_record.append(img_hash)
        new_samples_added += 1

    if new_samples_added == 0:
        print("\n[Continuous Learning] All images are already registered in the dataset.")
        print("[Continuous Learning] Skipping fine-tuning to prevent overfitting.")
        return

    with open(registry_file, "a") as f:
        for h in new_hashes_to_record:
            f.write(f"{h}\n")

    print(f"\n{'='*60}")
    print(f"CONTINUOUS LEARNING: Added {new_samples_added} new images. Retraining model...")
    print(f"{'='*60}")

    try:
        fine_tuner = YOLO(weights_path)
        fine_tuner.train(
            data=yaml_file,
            epochs=10,
            imgsz=640,
            batch=8,
            lr0=0.0005,
            name="lfia_continuous_finetune",
            exist_ok=True,
            verbose=False
        )

        latest_best = os.path.join(fine_tuner.trainer.save_dir, "weights", "best.pt")
        if os.path.exists(latest_best):
            shutil.copy(latest_best, weights_path)
            print(f"Model updated at: {weights_path}")
    except Exception as e:
        print(f"[Warning] Continuous fine-tuning skipped: {e}")
