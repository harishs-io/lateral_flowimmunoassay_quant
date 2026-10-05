"""
Automated Lateral Flow Immunoassay (LFIA) Quantification Pipeline
===================================================================
A high-throughput computer vision and bioanalytical pipeline for 
colloidal gold lateral flow assay quantification.

Key Features:
- YOLOv8-based automated membrane localization and snug cropping.
- LSPR optical channel differentiation (Green - Red) for colloidal gold (AuNP).
- Dynamic baseline-corrected peak integration (Control & Test lines).
- 1/y^2 weighted 4-Parameter Logistic (4PL) regression.
- CLSI EP17-compliant analytical limit determination (LoB & LoD).
- Automatic high-dose hook effect and membrane saturation plateau detection.
- Multi-channel optical absorbance justification reporting.
- MD5-hash protected continuous learning and model fine-tuning.

Usage:
------
# Basic run using default directory structure:
python lfia_analyzer.py

# Custom paths run via CLI arguments:
python lfia_analyzer.py --input_dir ./data/experiments \
                        --weights ./weights/best.pt \
                        --dataset_dir ./dataset
"""

import os
import glob
import shutil
import hashlib
import argparse
import warnings
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit, fsolve
from ultralytics import YOLO

# ---------------------------------------------------------
# PUBLICATION TYPOGRAPHY SETTINGS
# ---------------------------------------------------------
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Times', 'serif']
plt.rcParams['mathtext.fontset'] = 'stix'

warnings.filterwarnings("ignore")

# Backward compatibility for NumPy trapezoidal integration
integrate_area = getattr(np, "trapezoid", getattr(np, "trapz", None))

# ---------------------------------------------------------
# DEFAULT CONFIGURATION CONSTANTS
# ---------------------------------------------------------
MIN_PEAK_PROMINENCE = 3.0
MIN_PEAK_DISTANCE = 40
SHADOW_MAX_INTENSITY = 11.0
CONJUGATE_PAD_CUTOFF = 500
SMOOTH_DATA = False


def parse_arguments():
    """Parses command-line arguments with portable, relative defaults."""
    repo_root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Automated LFIA Signal Quantification and Calibration Pipeline"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=str(repo_root / "data"),
        help="Root folder containing assay subfolders with raw strip images."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=str(repo_root / "weights" / "best.pt"),
        help="Path to trained YOLOv8 model weights (.pt file)."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=str(repo_root / "dataset"),
        help="Directory containing data.yaml and image/label sets for fine-tuning."
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.20,
        help="YOLO detection confidence threshold."
    )
    return parser.parse_args()


# ---------------------------------------------------------
# ML AUTO-CROP FUNCTION (YOLO Object Detection)
# ---------------------------------------------------------
def auto_crop_strip(img_path, model, conf_thresh, sample_collector):
    """
    ML-based crop using fine-tuned YOLOv8.
    Passes image path directly to preserve color space, scale, and EXIF orientation.
    """
    raw_bgr = cv2.imread(img_path)
    if raw_bgr is None:
        raw_bgr = np.array(Image.open(img_path).convert('RGB'))[:, :, ::-1]

    img_h, img_w = raw_bgr.shape[:2]

    results = model.predict(source=img_path, conf=conf_thresh, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    if len(boxes) > 0:
        best_box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        x1, y1, x2, y2 = map(int, best_box[:4])

        sample_collector.append((img_path, x1, y1, x2, y2, img_w, img_h))

        x_start = max(0, x1)
        x_end = min(img_w, x2)
        y_start = max(0, y1)
        y_end = min(img_h, y2)

        cropped_bgr = raw_bgr[y_start:y_end, x_start:x_end]
        return cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)

    return cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)


def clean_concentration(val):
    """Extracts numeric concentration from filename metadata."""
    val_str = str(val).strip().lower()
    if 'blank' in val_str:
        return 0.0
    match = re.search(r'[\d.]+', val_str)
    if match:
        return float(match.group())
    return np.nan


def logistic4(x, a, b, c, d):
    """Four-Parameter Logistic (4PL) regression model."""
    return d + (a - d) / (1.0 + (x / np.maximum(c, 1e-9))**b)


# ---------------------------------------------------------
# CONTINUOUS LEARNING / FINE-TUNING MODULE
# ---------------------------------------------------------
def get_file_hash(filepath):
    """Calculates MD5 hash of an image file to detect duplicate content."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()


def fine_tune_model(dataset_dir, weights_path, collected_samples):
    """
    Safely fine-tunes the YOLO model only when genuinely new images are processed.
    Uses MD5 checksums to prevent re-training on previously seen images.
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

        # Normalized YOLO format coordinates
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
        print("\n[Continuous Learning] All processed images are already registered.")
        print("[Continuous Learning] Skipping fine-tuning to prevent model overfitting.")
        return

    with open(registry_file, "a") as f:
        for h in new_hashes_to_record:
            f.write(f"{h}\n")

    print(f"\n{'='*60}")
    print(f"CONTINUOUS LEARNING: Added {new_samples_added} new images. Fine-tuning model...")
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
            print(f"Model successfully fine-tuned and updated at: {weights_path}")
    except Exception as e:
        print(f"[Warning] Continuous fine-tuning skipped due to error: {e}")


# ---------------------------------------------------------
# MASTER PIPELINE EXECUTION
# ---------------------------------------------------------
def run_pipeline(base_dir, dataset_dir, weights_path, conf_thresh):
    """Executes multi-folder LFIA batch processing and statistical quantification."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Model weights not found at '{weights_path}'. "
            "Please ensure weights are trained or specified via --weights."
        )

    ml_detector = YOLO(weights_path)
    collected_samples = []

    subfolders = [f.path for f in os.scandir(base_dir) if f.is_dir() and "Results" not in f.path]
    if not subfolders:
        subfolders = [base_dir]

    for folder in subfolders:
        folder_name = os.path.basename(folder)
        print(f"\n{'='*60}")
        print(f"ANALYZING FOLDER: {folder_name}")
        print(f"{'='*60}")

        extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(folder, ext)))

        if not image_files:
            print(f"No LFA images found in '{folder_name}'. Skipping...")
            continue

        image_files.sort(reverse=True)

        file_names, concentrations, replicate_numbers = [], [], []
        t_areas, c_areas, tc_ratios = [], [], []
        r_peaks, g_peaks, b_peaks, diff_peaks = [], [], [], []

        results_dir = os.path.join(folder, "Results")
        os.makedirs(results_dir, exist_ok=True)

        substrate_name = "AuNP"

        for img_path in image_files:
            file_name = os.path.basename(img_path)
            file_info = os.path.splitext(file_name)[0].split('_')
            concentration = str(file_info[0])
            replicate_number = str(file_info[1]) if len(file_info) > 1 else ""
            if len(file_info) > 2:
                substrate_name = str(file_info[2])

            print(f"\rProcessing: {concentration:<25}", end="", flush=True)

            conc_dir = os.path.join(results_dir, concentration.strip())
            os.makedirs(conc_dir, exist_ok=True)

            img = auto_crop_strip(img_path, ml_detector, conf_thresh, collected_samples)
            Image.fromarray(img).save(os.path.join(conc_dir, f"{file_name}_cropped_original.png"))

            # -----------------------------------------------------
            # MULTI-CHANNEL ABSORBANCE EXTRACTION
            # -----------------------------------------------------
            red_channel = img[:, :, 0]
            green_channel = img[:, :, 1]
            blue_channel = img[:, :, 2]

            red_average = 255.0 - np.mean(red_channel, axis=1)
            green_average = 255.0 - np.mean(green_channel, axis=1)
            blue_average = 255.0 - np.mean(blue_channel, axis=1)

            # Differential AuNP LSPR signal
            color_specific_signal = green_average - red_average
            best_channel = np.clip(color_specific_signal, 0, 255)
            best_channel[best_channel < 5] = 0

            if SMOOTH_DATA:
                window_size = 5
                best_channel = np.convolve(best_channel, np.ones(window_size) / window_size, mode='same')

            signal_length = len(best_channel)
            top_cutoff = int(signal_length * 0.15)
            bottom_cutoff = int(signal_length * 0.85)
            best_channel[:top_cutoff] = 0
            best_channel[bottom_cutoff:] = 0

            # 1. Peak Detection
            raw_peaks, properties = find_peaks(
                best_channel,
                distance=MIN_PEAK_DISTANCE,
                prominence=MIN_PEAK_PROMINENCE,
                width=1
            )

            sort_order = np.argsort(raw_peaks)
            sorted_peaks = list(raw_peaks[sort_order])
            sorted_prominences = list(properties['prominences'][sort_order])

            # 2. Absorbent Pad Shadow Filter
            if len(sorted_peaks) > 0:
                first_peak = sorted_peaks[0]
                if best_channel[first_peak] <= SHADOW_MAX_INTENSITY:
                    sorted_peaks.pop(0)
                    sorted_prominences.pop(0)

            # 3. Conjugate Pad Cutoff Filter
            candidate_peaks = [p for p in sorted_peaks if p <= CONJUGATE_PAD_CUTOFF]
            candidate_proms = [
                sorted_prominences[i]
                for i, p in enumerate(sorted_peaks)
                if p <= CONJUGATE_PAD_CUTOFF
            ]

            # 4. Spatial Anchor / Pair Matching
            final_peaks = []
            if len(candidate_peaks) > 0:
                valid_pairs = []
                for i in range(len(candidate_peaks)):
                    for j in range(i + 1, len(candidate_peaks)):
                        if 45 <= (candidate_peaks[j] - candidate_peaks[i]) <= 90:
                            valid_pairs.append((candidate_peaks[i], candidate_peaks[j]))

                if len(valid_pairs) > 0:
                    best_pair = max(
                        valid_pairs,
                        key=lambda pair: best_channel[pair[0]] + best_channel[pair[1]]
                    )
                    final_peaks = [best_pair[0], best_pair[1]]
                else:
                    strongest_idx = np.argmax(candidate_proms)
                    final_peaks = [candidate_peaks[strongest_idx]]

            final_peaks = np.array(final_peaks)

            # 5. Extract Optical Density at Peak Anchor
            anchor_idx = final_peaks[0] if len(final_peaks) > 0 else np.argmax(green_average)
            r_val = float(red_average[anchor_idx])
            g_val = float(green_average[anchor_idx])
            b_val = float(blue_average[anchor_idx])
            diff_val = float(g_val - r_val)

            r_peaks.append(r_val)
            g_peaks.append(g_val)
            b_peaks.append(b_val)
            diff_peaks.append(diff_val)

            # 6. Sloped Dynamic Baseline Area Integration
            c_area, t_area = 0.0, 0.0
            x_c_baseline, y_c_baseline = [], []
            x_t_baseline, y_t_baseline = [], []
            x_control_begin, x_control_end = 0, 0
            x_test_begin, x_test_end = 0, 0

            if len(final_peaks) > 0:
                widths_half, _, _, _ = peak_widths(best_channel, final_peaks, rel_height=0.5)

                # Control Line Integration
                c_idx = final_peaks[0]
                w_c = widths_half[0]
                x_control_begin = max(0, int(round(c_idx - 2 * w_c)))
                x_control_end = min(len(best_channel) - 1, int(round(c_idx + 2 * w_c)))
                y_control_begin = best_channel[x_control_begin]
                y_control_end = best_channel[x_control_end]

                slope_c = (
                    (y_control_end - y_control_begin) / (x_control_end - x_control_begin)
                    if (x_control_end - x_control_begin) != 0 else 0
                )
                x_c_baseline = np.arange(x_control_begin, x_control_end + 1)
                y_c_baseline = y_control_end + (x_c_baseline - x_control_end) * slope_c

                for i, x in enumerate(x_c_baseline):
                    if best_channel[x] < y_c_baseline[i]:
                        best_channel[x] = y_c_baseline[i]

                c_area = (
                    integrate_area(best_channel[x_control_begin:x_control_end + 1], x_c_baseline)
                    - integrate_area([y_control_begin, y_control_end], [x_control_begin, x_control_end])
                )

                # Test Line Integration
                if len(final_peaks) > 1:
                    t_idx = final_peaks[1]
                    w_t = widths_half[1]
                    x_test_begin = max(0, int(round(t_idx - 2 * w_t)))
                    x_test_end = min(len(best_channel) - 1, int(round(t_idx + 2 * w_t)))
                    y_test_begin = best_channel[x_test_begin]
                    y_test_end = best_channel[x_test_end]

                    slope_t = (
                        (y_test_end - y_test_begin) / (x_test_end - x_test_begin)
                        if (x_test_end - x_test_begin) != 0 else 0
                    )
                    x_t_baseline = np.arange(x_test_begin, x_test_end + 1)
                    y_t_baseline = y_test_end + (x_t_baseline - x_test_end) * slope_t

                    for i, x in enumerate(x_t_baseline):
                        if best_channel[x] < y_t_baseline[i]:
                            best_channel[x] = y_t_baseline[i]

                    t_area = (
                        integrate_area(best_channel[x_test_begin:x_test_end + 1], x_t_baseline)
                        - integrate_area([y_test_begin, y_test_end], [x_test_begin, x_test_end])
                    )

            tc_ratio = t_area / c_area if c_area != 0 else 0.0

            # Write Individual Strip Result File
            with open(os.path.join(conc_dir, f"{file_name}_results.txt"), "w") as text_file:
                text_file.write(f"File Name: {file_name}\n")
                text_file.write(f"Concentration: {concentration}\n")
                text_file.write(f"Replicate: {replicate_number}\n")
                text_file.write("-" * 35 + "\n")
                text_file.write(f"Control (C) Area: {c_area:.4f}\n")
                text_file.write(f"Test (T) Area: {t_area:.4f}\n")
                text_file.write(f"T/C Ratio: {tc_ratio:.6f}\n")
                text_file.write("-" * 35 + "\n")
                text_file.write("RAW OPTICAL ABSORBANCE AT BAND PEAK:\n")
                text_file.write(f"Green Channel (530 nm Absorbance): {g_val:.2f} a.u.\n")
                text_file.write(f"Red Channel   (650 nm Reflection): {r_val:.2f} a.u.\n")
                text_file.write(f"Blue Channel  (450 nm Absorbance): {b_val:.2f} a.u.\n")
                text_file.write(f"Differential Signal (Green - Red): {diff_val:.2f} a.u.\n")

            # Publication Profile Plot Generation
            fig = plt.figure(figsize=(9.5, 4.8), dpi=300)
            gs = fig.add_gridspec(
                2, 2, width_ratios=[1.2, 5.0], height_ratios=[4.0, 1.2],
                wspace=0.18, hspace=0.10
            )

            ax1 = fig.add_subplot(gs[0, 0])
            ax_leg = fig.add_subplot(gs[1, 0])
            ax2 = fig.add_subplot(gs[:, 1])

            ax1.imshow(img)
            clean_conc_text = re.sub(r'(?i)\s*ng(/ml)?', '', str(concentration).strip())
            title_disp = f"{clean_conc_text} ng/mL" + (f"\n({replicate_number})" if replicate_number else "")
            ax1.set_title(title_disp, fontsize=9.5, weight='bold', pad=6)
            ax1.axis('off')

            line_signal, = ax2.plot(best_channel, color='#1b7837', linewidth=1.8, label='AuNP Signal (G - R)')
            ax2.set_xlabel('Migration Distance (pixels)', fontsize=10, fontweight='semibold', labelpad=6)
            ax2.set_ylabel('Inverted Absorbance (a.u.)', fontsize=10, fontweight='semibold', labelpad=6)
            ax2.tick_params(direction='out', length=4, width=1, labelsize=9.5)

            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['left'].set_linewidth(1.0)
            ax2.spines['bottom'].set_linewidth(1.0)

            y_max_profile = max(10, np.max(best_channel) * 1.30)
            ax2.set_ylim(-0.5, y_max_profile)
            ax2.set_xlim(0, len(best_channel))

            legend_handles = [line_signal]

            if len(final_peaks) > 0:
                c_peak_x = final_peaks[0]
                c_peak_y = best_channel[c_peak_x]
                ax2.plot(x_c_baseline, y_c_baseline, color='#762a83', linestyle='--', linewidth=1.2)
                fill_c = ax2.fill_between(
                    x_c_baseline, best_channel[x_control_begin:x_control_end + 1],
                    y_c_baseline, color='#af8dc3', alpha=0.55,
                    label=f'C Line ({c_area:.1f})'
                )
                legend_handles.append(fill_c)
                ax2.annotate(
                    'Control Line',
                    xy=(c_peak_x, c_peak_y),
                    xytext=(c_peak_x, c_peak_y + (y_max_profile * 0.08)),
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#40004b',
                    arrowprops=dict(arrowstyle='->', color='#762a83', lw=1.0, shrinkA=2, shrinkB=2)
                )

            if len(final_peaks) > 1:
                t_peak_x = final_peaks[1]
                t_peak_y = best_channel[t_peak_x]
                ax2.plot(x_t_baseline, y_t_baseline, color='#2166ac', linestyle='--', linewidth=1.2)
                fill_t = ax2.fill_between(
                    x_t_baseline, best_channel[x_test_begin:x_test_end + 1],
                    y_t_baseline, color='#92c5de', alpha=0.55,
                    label=f'T Line ({t_area:.1f})'
                )
                legend_handles.append(fill_t)
                ax2.annotate(
                    'Test Line',
                    xy=(t_peak_x, t_peak_y),
                    xytext=(t_peak_x, t_peak_y + (y_max_profile * 0.08)),
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#053061',
                    arrowprops=dict(arrowstyle='->', color='#2166ac', lw=1.0, shrinkA=2, shrinkB=2)
                )

            ax_leg.axis('off')
            ax_leg.legend(
                handles=legend_handles,
                loc='upper center',
                bbox_to_anchor=(0.5, 1.1),
                fontsize=7.5,
                frameon=True,
                facecolor='#fafafa',
                edgecolor='#d9d9d9',
                handletextpad=0.4,
                borderpad=0.5,
                labelspacing=0.4
            )

            stats_text = f"T/C Ratio: {tc_ratio:.4f}"
            ax2.text(
                0.97, 0.95, stats_text, transform=ax2.transAxes,
                fontsize=9.5, fontweight='bold', verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.35', facecolor='#f7f7f7', edgecolor='#cccccc', alpha=0.9)
            )

            plt.savefig(os.path.join(conc_dir, f"{file_name}_Analysis_Plot.png"), bbox_inches='tight', dpi=300)
            plt.close(fig)

            file_names.append(file_name)
            concentrations.append(concentration)
            replicate_numbers.append(replicate_number)
            t_areas.append(t_area)
            c_areas.append(c_area)
            tc_ratios.append(tc_ratio)

        print(f"\rProcessing: Completed!{' ' * 20}\n")

        # ---------------------------------------------------------
        # STATISTICAL ANALYSIS & HOOK-AWARE 4PL CALIBRATION
        # ---------------------------------------------------------
        results_df = pd.DataFrame({
            'File Name': file_names,
            'Concentration (ng/mL)': concentrations,
            'Replicate Number': replicate_numbers,
            'T': t_areas,
            'C': c_areas,
            'T/C': tc_ratios,
            'Green_Abs': g_peaks,
            'Red_Abs': r_peaks,
            'Blue_Abs': b_peaks,
            'Diff_Signal': diff_peaks
        })

        results_df['Concentration Numeric'] = results_df['Concentration (ng/mL)'].apply(clean_concentration)
        results_df.dropna(subset=['Concentration Numeric'], inplace=True)

        grouped = results_df.groupby('Concentration Numeric')['T/C'].agg(['mean', 'std']).reset_index()
        grouped.rename(columns={'mean': 'Mean T/C', 'std': 'Standard Deviation T/C'}, inplace=True)
        grouped['CV (%)'] = (grouped['Standard Deviation T/C'] / grouped['Mean T/C']) * 100
        grouped.fillna(0, inplace=True)

        print(f"Stats for {folder_name}:")
        print(grouped.to_string(index=True))

        conc_values = grouped['Concentration Numeric'].values
        avg_tc = grouped['Mean T/C'].values
        std_tc = grouped['Standard Deviation T/C'].values

        lod_result_text = ""
        fit_result_text = ""
        hook_result_text = ""

        if len(avg_tc) > 3:
            conc_fit = conc_values.copy()
            if len(conc_fit) > 0 and conc_fit[0] == 0:
                non_zero_min = np.min(conc_fit[conc_fit > 0]) if np.any(conc_fit > 0) else 0.01
                conc_fit[0] = non_zero_min / 10.0

            # 1. High-Dose Hook Effect & Saturation Detection
            max_idx = np.argmax(avg_tc)
            hook_conc = conc_values[max_idx]
            hook_signal = avg_tc[max_idx]

            is_hook = False
            if max_idx < len(avg_tc) - 1 and avg_tc[max_idx] > avg_tc[-1] * 1.05:
                is_hook = True
                fit_mask = np.arange(len(conc_fit)) <= (max_idx + 1)
                hook_result_text = f"High-Dose Hook Effect detected at > {hook_conc:.2f} ng/mL (Peak T/C = {hook_signal:.3f})"
            else:
                fit_mask = np.ones(len(conc_fit), dtype=bool)
                hook_result_text = f"Membrane Saturation Plateau reached at >= {hook_conc:.2f} ng/mL (Max T/C = {hook_signal:.3f})"

            fit_x = conc_fit[fit_mask]
            fit_y = avg_tc[fit_mask]

            # 2. Inverse-Variance 1/y^2 Weighted 4PL Curve Fitting
            p0 = [np.min(fit_y), 1.0, np.median(fit_x), np.max(fit_y)]
            lower_bounds = [0.0, 0.1, fit_x.min() * 0.1, np.min(fit_y)]
            upper_bounds = [np.max(fit_y), 5.0, fit_x.max() * 5.0, np.max(fit_y) * 2.0]

            weights = 1.0 / np.maximum(fit_y, 0.05)**2
            sigma_weights = 1.0 / np.sqrt(weights)

            try:
                popt, _ = curve_fit(
                    logistic4,
                    fit_x,
                    fit_y,
                    p0=p0,
                    bounds=(lower_bounds, upper_bounds),
                    sigma=sigma_weights,
                    method='trf',
                    maxfev=20000
                )
                a, b, c, d = popt

                residuals = fit_y - logistic4(fit_x, *popt)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((fit_y - np.mean(fit_y))**2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                fit_result_text = (
                    f"4PL Parameters (Dynamic Fit): a={a:.4f}, b={b:.4f}, "
                    f"c={c:.4f}, d={d:.4f} | R^2 = {r_squared:.4f}"
                )

                # 3. CLSI EP17 Limit of Blank (LoB) & Limit of Detection (LoD)
                detectable_indices = avg_tc != 0
                if np.any(detectable_indices):
                    mu_blank = avg_tc[0]
                    sigma_blank = std_tc[0] if std_tc[0] > 0 else 0.01
                    sigma_low = std_tc[1] if len(std_tc) > 1 and std_tc[1] > 0 else sigma_blank

                    lob_signal = mu_blank + 1.645 * sigma_blank
                    lod_signal = lob_signal + 1.645 * sigma_low

                    def solve_lod(x):
                        return logistic4(x, a, b, c, d) - lod_signal

                    lod_concentration = fsolve(solve_lod, x0=fit_x.min())[0]

                    if fit_x.min() * 0.1 <= lod_concentration <= c:
                        lod_result_text = (
                            f"Calculated LoD = {lod_concentration:.4f} ng/mL "
                            f"(LoD Signal = {lod_signal:.4f}, LoB Signal = {lob_signal:.4f})"
                        )
                    else:
                        lod_result_text = (
                            f"Analytical LoD Signal ({lod_signal:.4f}) falls outside dynamic binding range."
                        )

                # 4. Standard Curve Plotting
                fig_curve, ax = plt.subplots(figsize=(6.5, 5.2), dpi=300)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_linewidth(1.0)
                ax.spines['bottom'].set_linewidth(1.0)
                ax.tick_params(direction='out', length=5, width=1.0, labelsize=10, which='major')
                ax.tick_params(direction='out', length=3, width=0.8, which='minor')

                ax.errorbar(
                    conc_values, avg_tc, yerr=std_tc, fmt='o',
                    color='#0f2a4a', ecolor='#4a5568', elinewidth=1.2, capsize=3.5, capthick=1.0,
                    markersize=6, markerfacecolor='#1d4ed8', markeredgewidth=1.0,
                    label="Experimental Data (Mean ± SD)", zorder=3
                )

                x_smooth = np.logspace(np.log10(conc_fit.min() * 0.5), np.log10(conc_fit.max() * 2.0), 350)
                ax.plot(
                    x_smooth, logistic4(x_smooth, *popt), color='#dc2626', linewidth=1.8,
                    label=f"4PL Fit ($R^2$ = {r_squared:.4f})", zorder=2
                )

                if 'lod_concentration' in locals() and fit_x.min() * 0.1 <= lod_concentration <= c:
                    ax.plot(
                        lod_concentration, lod_signal, marker='D', markersize=6.5,
                        color='#111827', markerfacecolor='#111827', zorder=4,
                        label=f"LoD: {lod_concentration:.4f} ng/mL"
                    )
                    x_axis_min = conc_fit.min() * 0.5
                    ax.hlines(lod_signal, xmin=x_axis_min, xmax=lod_concentration,
                              color='#6b7280', linestyle='--', linewidth=1.0, zorder=1)
                    ax.vlines(lod_concentration, ymin=0, ymax=lod_signal,
                              color='#6b7280', linestyle='--', linewidth=1.0, zorder=1)

                zone_label = "Hook Inhibition Zone" if is_hook else "Saturation Plateau"
                ax.axvspan(hook_conc, conc_fit.max() * 2.5, color='#fef3c7', alpha=0.45, label=zone_label)

                ax.set_xscale('log')
                ax.set_xlabel("Target Concentration (ng/mL)", fontsize=11, fontweight='bold', labelpad=7)
                ax.set_ylabel("Signal Ratio (T/C)", fontsize=11, fontweight='bold', labelpad=7)

                title_str = substrate_name.split('.')[0] if substrate_name else "AuNP LFIA"
                ax.set_title(f"{title_str} Standard Curve (Hook Corrected)\n{folder_name}", fontsize=11.5, fontweight='bold', pad=12)

                ax.set_xlim(conc_fit.min() * 0.5, conc_fit.max() * 2.0)
                y_upper = max(np.max(avg_tc) + np.max(std_tc), d if 'd' in locals() else 1.0) * 1.15
                ax.set_ylim(0, y_upper)

                ax.legend(
                    frameon=True, facecolor='#ffffff', edgecolor='#e2e8f0', fontsize=8,
                    loc='lower right', borderpad=0.4, handletextpad=0.4, labelspacing=0.35, handlelength=1.4
                )

                plt.savefig(os.path.join(results_dir, f"{folder_name}_Standard_Curve_Plot.png"), bbox_inches='tight', dpi=300)
                plt.close(fig_curve)

                if lod_result_text:
                    if "Calculated LoD" in lod_result_text:
                        print(f"Calculated LoD = {lod_concentration:.4f} ng/mL")
                    else:
                        print(lod_result_text)

            except Exception as e:
                lod_result_text = f"Could not fit curve for {folder_name}: {e}"
                print(lod_result_text)

        else:
            lod_result_text = f"Not enough distinct data points to plot the standard curve for {folder_name}."
            print(lod_result_text)

        plt.close('all')

        # ---------------------------------------------------------
        # OPTICAL CHANNEL JUSTIFICATION STATS
        # ---------------------------------------------------------
        mean_g = np.mean(g_peaks) if g_peaks else 0.0
        mean_r = np.mean(r_peaks) if r_peaks else 0.0
        mean_b = np.mean(b_peaks) if b_peaks else 0.0
        mean_diff = np.mean(diff_peaks) if diff_peaks else 0.0
        contrast_ratio = (mean_g / mean_r) if mean_r > 0 else 1.0

        # Master Summary Text Report
        summary_file_path = os.path.join(results_dir, f"{folder_name}_Summary_Results.txt")
        with open(summary_file_path, "w") as sf:
            sf.write("=" * 75 + "\n")
            sf.write(f"ANALYSIS SUMMARY: {folder_name}\n")
            sf.write("=" * 75 + "\n\n")

            sf.write("1. STATISTICAL SUMMARY ACROSS CONCENTRATIONS:\n")
            sf.write("-" * 75 + "\n")
            sf.write(grouped.to_string(index=True) + "\n\n")

            sf.write("2. ANALYTICAL LIMITS & MODEL FIT (HOOK-EFFECT CORRECTED):\n")
            sf.write("-" * 75 + "\n")
            if hook_result_text:
                sf.write(f" * {hook_result_text}\n")
            if lod_result_text:
                sf.write(f" * {lod_result_text}\n")
            if fit_result_text:
                sf.write(f" * {fit_result_text}\n")
            sf.write("\n")

            sf.write("3. OPTICAL CHANNEL SELECTION JUSTIFICATION (COLLOIDAL GOLD / AuNP):\n")
            sf.write("-" * 75 + "\n")
            sf.write(f" * Mean Green Absorbance (A_G at 530 nm) : {mean_g:.2f} a.u.  [MAXIMUM ABSORPTION]\n")
            sf.write(f" * Mean Blue Absorbance  (A_B at 450 nm) : {mean_b:.2f} a.u.  [INTERMEDIATE]\n")
            sf.write(f" * Mean Red Absorbance   (A_R at 650 nm) : {mean_r:.2f} a.u.  [MINIMAL / REFLECTED]\n")
            sf.write(f" * Differential Contrast (Green - Red)   : {mean_diff:.2f} a.u.  [ACTIVE ANALYTICAL SIGNAL]\n")
            sf.write(f" * Green-to-Red Absorption Contrast Ratio: {contrast_ratio:.2f}x\n")
            sf.write(" CONCLUSION: Green channel absorbance is maximal due to localized surface plasmon\n")
            sf.write(" resonance (LSPR) of AuNP. Subtraction of the red channel effectively suppresses\n")
            sf.write(" background membrane scattering and illumination drift.\n\n")

            sf.write("4. INDIVIDUAL REPLICATE READINGS (RAW CHANNELS & INTEGRATED AREAS):\n")
            sf.write("-" * 75 + "\n")
            display_cols = ['File Name', 'Concentration (ng/mL)', 'Replicate Number', 'C', 'T', 'T/C',
                            'Green_Abs', 'Red_Abs', 'Diff_Signal']
            sf.write(results_df[display_cols].to_string(index=False) + "\n")

        print(f"Summary saved: {summary_file_path}")

    print("\n=========================================================")
    print("ALL FOLDERS PROCESSED SUCCESSFULLY!")
    print("=========================================================")

    # ---------------------------------------------------------
    # INTERACTIVE USER PROMPT FOR MODEL FINE-TUNING
    # ---------------------------------------------------------
    if collected_samples and os.path.exists(dataset_dir):
        user_choice = input(
            "\nDo you want to fine-tune the YOLO model with the newly processed data? (y/n): "
        ).strip().lower()
        if user_choice in ['y', 'yes']:
            fine_tune_model(dataset_dir, weights_path, collected_samples)
        else:
            print("Model fine-tuning skipped. Model weights kept frozen for reproducibility.")


if __name__ == "__main__":
    cli_args = parse_arguments()
    run_pipeline(
        base_dir=cli_args.input_dir,
        dataset_dir=cli_args.dataset_dir,
        weights_path=cli_args.weights,
        conf_thresh=cli_args.conf
    )