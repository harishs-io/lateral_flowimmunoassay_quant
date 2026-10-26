import os
import sys
from pathlib import Path

# Set cache directories and environment variables
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mpl_cache')
os.environ.setdefault('YOLO_CONFIG_DIR', '/tmp/yolo_config')

# Ensure parent directory is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import glob
import argparse
import warnings
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from ultralytics import YOLO

from src.utils import clean_concentration
from src.detector import auto_crop_strip, detect_bands_stage2, fine_tune_model
from src.signal_processing import (
    extract_lspr_signal,
    extract_longitudinal_baseline,
    integrate_band_adaptive,
    MIN_PEAK_PROMINENCE,
    MIN_SADDLE_DELTA
)
from src.calibration import (
    logistic4,
    detect_hook_and_saturation,
    fit_weighted_4pl,
    calculate_clsi_limits
)

# Typography settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Times', 'serif']
plt.rcParams['mathtext.fontset'] = 'stix'
warnings.filterwarnings("ignore")

def parse_args():
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="LFIA Two-Stage Cascaded Quantification Pipeline")
    parser.add_argument("--input_dir", type=str, default=str(repo_root / "demo_data"),
                        help="Root folder containing assay subdirectories.")
    parser.add_argument("--strip_weights", type=str, default=str(repo_root / "weights" / "best_strip_detector.pt"),
                        help="Path to Stage 1 Strip YOLOv8 weights (.pt).")
    parser.add_argument("--band_weights", type=str, default=str(repo_root / "weights" / "best_band_detector.pt"),
                        help="Path to Stage 2 Band YOLOv8 weights (.pt).")
    parser.add_argument("--weights", type=str, default=None,
                        help="Legacy single weights fallback argument.")
    parser.add_argument("--dataset_dir", type=str, default=str(repo_root / "dataset"),
                        help="Root directory for continuous learning dataset.")
    parser.add_argument("--strip_conf", type=float, default=0.20, help="Stage 1 strip detector confidence.")
    parser.add_argument("--band_conf", type=float, default=0.25, help="Stage 2 band detector confidence.")
    return parser.parse_args()

def run_pipeline(base_dir, dataset_dir=None, strip_weights=None, band_weights=None,
                 strip_conf=0.20, band_conf=0.25, weights_path=None, conf_thresh=None):
    """
    Executes the Two-Stage Cascaded LFIA Signal Quantification Pipeline.
    Stage 1: Strip localization & auto-crop.
    Stage 2: Sub-pixel C & T band detection + LSPR AUC integration + 4PL calibration.
    """
    repo_root = Path(__file__).resolve().parent.parent

    # Handle backward-compatible argument aliases
    if conf_thresh is not None:
        strip_conf = conf_thresh
        band_conf = max(conf_thresh, 0.25)

    if strip_weights is None:
        if weights_path is not None and os.path.exists(weights_path):
            strip_weights = weights_path
        else:
            default_strip = str(repo_root / "weights" / "best_strip_detector.pt")
            strip_weights = default_strip if os.path.exists(default_strip) else str(repo_root / "weights" / "best.pt")

    if band_weights is None:
        default_band = str(repo_root / "weights" / "best_band_detector.pt")
        band_weights = default_band if os.path.exists(default_band) else str(repo_root / "weights" / "best.pt")

    if not os.path.exists(strip_weights):
        raise FileNotFoundError(f"Strip detector weights not found: '{strip_weights}'")
    if not os.path.exists(band_weights):
        raise FileNotFoundError(f"Band detector weights not found: '{band_weights}'")

    print(f"Loading Stage 1 Strip Detector: {strip_weights}")
    strip_detector = YOLO(strip_weights)
    print(f"Loading Stage 2 Band Detector : {band_weights}")
    band_detector = YOLO(band_weights)

    collected_samples = []

    subfolders = [f.path for f in os.scandir(base_dir) if f.is_dir() and "Results" not in f.path]
    if not subfolders:
        subfolders = [base_dir]

    execution_summary = {}

    for folder in subfolders:
        folder_name = os.path.basename(folder)
        print(f"\n{'='*65}\nANALYZING FOLDER: {folder_name}\n{'='*65}")

        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(glob.glob(os.path.join(folder, ext)))

        if not image_files:
            print(f"No LFIA images found in '{folder_name}'. Skipping...")
            continue

        image_files.sort(reverse=True)
        results_dir = os.path.join(folder, "Results")
        os.makedirs(results_dir, exist_ok=True)

        # ---------------------------------------------------------
        # PASS 1: BATCH GEOMETRY REGISTRATION
        # ---------------------------------------------------------
        print("Pass 1: Auto-registering batch geometry & retention factors...", flush=True)
        batch_c_positions = []
        batch_t_positions = []
        batch_delta_y = []
        batch_c_widths = []
        batch_t_widths = []
        cached_crops = {}

        for img_p in image_files:
            crop = auto_crop_strip(img_p, strip_detector, conf_thresh=strip_conf, sample_collector=collected_samples)
            cached_crops[img_p] = crop
            c_test, t_test = detect_bands_stage2(crop, band_detector, conf_thresh=band_conf)

            if c_test is not None:
                batch_c_positions.append(c_test['yc'])
                batch_c_widths.append(c_test['y2'] - c_test['y1'])
            if t_test is not None:
                batch_t_positions.append(t_test['yc'])
                batch_t_widths.append(t_test['y2'] - t_test['y1'])
            if c_test is not None and t_test is not None:
                batch_delta_y.append(t_test['yc'] - c_test['yc'])

        median_c_y = float(np.median(batch_c_positions)) if batch_c_positions else 120.0
        median_t_y = float(np.median(batch_t_positions)) if batch_t_positions else 155.0
        median_delta_y = float(np.median(batch_delta_y)) if batch_delta_y else max(25.0, median_t_y - median_c_y)
        median_c_width = float(np.median(batch_c_widths)) if batch_c_widths else 10.0
        median_t_width = float(np.median(batch_t_widths)) if batch_t_widths else 16.0

        print(f"Registered Assay Layout: C @ {median_c_y:.1f} px | T @ {median_c_y + median_delta_y:.1f} px (Spacing: +{median_delta_y:.1f} px)\n")

        # ---------------------------------------------------------
        # PASS 2: FULL SIGNAL QUANTIFICATION
        # ---------------------------------------------------------
        file_names, concentrations, replicate_nums = [], [], []
        t_areas, c_areas, tc_ratios = [], [], []
        c_confs, t_confs, det_sources = [], [], []
        r_peaks, g_peaks, b_peaks, diff_peaks = [], [], [], []
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

            img = cached_crops[img_path]
            Image.fromarray(img).save(os.path.join(conc_dir, f"{file_name}_cropped_original.png"))

            # Differential LSPR extraction (G - R absorbance)
            raw_signal, red_avg, green_avg, blue_avg, red_raw, green_raw, blue_raw = extract_lspr_signal(img)
            sig_len = len(raw_signal)

            membrane_baseline = extract_longitudinal_baseline(raw_signal, filter_radius=35)
            clean_channel = np.zeros_like(raw_signal)

            c_det, t_det = detect_bands_stage2(img, band_detector, conf_thresh=band_conf)

            detection_source = "Stage 2 YOLO"
            c_conf_val = float(c_det['conf']) if c_det else 0.0
            t_conf_val = float(t_det['conf']) if t_det else 0.0

            # 1D topological fallback if Stage 2 missed C-line
            if c_det is None:
                c_search_win = raw_signal[max(0, int(median_c_y - 25)):min(sig_len, int(median_c_y + 25))]
                if len(c_search_win) > 3 and np.max(c_search_win) > 3.0:
                    c_fallback_y = max(0, int(median_c_y - 25)) + int(np.argmax(c_search_win))
                    c_det = {
                        'yc': float(c_fallback_y),
                        'y1': float(c_fallback_y - median_c_width / 2.0),
                        'y2': float(c_fallback_y + median_c_width / 2.0),
                        'conf': 0.50
                    }
                    detection_source = "Hybrid 1D Topological Fallback"

            final_peaks = []
            C_area, T_area = 0.0, 0.0
            x_c_baseline, x_t_baseline = [], []
            x_c_begin, x_c_end = 0, 0
            x_t_begin, x_t_end = 0, 0

            # Control line integration
            if c_det is not None:
                apex_c, l_c, r_c, xs_c, p_c, C_area, raw_auc_c, base_auc_c, raw_pk_c, net_pk_c = integrate_band_adaptive(
                    raw_signal, membrane_baseline, c_det['yc'], expected_width=median_c_width,
                    membrane_height=sig_len, rel_threshold=0.05, min_prominence=MIN_PEAK_PROMINENCE
                )
                if C_area > 0.0:
                    final_peaks.append(apex_c)
                    clean_channel[l_c:r_c + 1] = p_c
                    x_c_begin, x_c_end = l_c, r_c
                    x_c_baseline = xs_c

                # Test line integration
                t_integrated = False
                if t_det is not None:
                    apex_t, l_t, r_t, xs_t, p_t, T_cand, raw_auc_t, base_auc_t, raw_pk_t, net_pk_t = integrate_band_adaptive(
                        raw_signal, membrane_baseline, t_det['yc'], expected_width=median_t_width,
                        membrane_height=sig_len, rel_threshold=0.05, min_prominence=MIN_PEAK_PROMINENCE
                    )
                    if T_cand > 0.0:
                        T_area = T_cand
                        final_peaks.append(apex_t)
                        clean_channel[l_t:r_t + 1] = p_t
                        x_t_begin, x_t_end = l_t, r_t
                        x_t_baseline = xs_t
                        t_integrated = True

                if not t_integrated:
                    # Virtual T-line Fallback using registered batch spacing
                    v_t_center = c_det['yc'] + median_delta_y
                    apex_t, l_t, r_t, xs_t, p_t, T_cand, raw_auc_t, base_auc_t, raw_pk_t, net_pk_t = integrate_band_adaptive(
                        raw_signal, membrane_baseline, v_t_center, expected_width=median_t_width,
                        membrane_height=sig_len, rel_threshold=0.05, min_prominence=MIN_PEAK_PROMINENCE
                    )
                    if T_cand > 0.0:
                        T_area = T_cand
                        final_peaks.append(apex_t)
                        clean_channel[l_t:r_t + 1] = p_t
                        x_t_begin, x_t_end = l_t, r_t
                        x_t_baseline = xs_t
                        detection_source = "Stage 2 (Virtual T Spatial Fallback)"
                    else:
                        T_area = 0.0
                        x_t_begin, x_t_end = 0, 0
                        x_t_baseline = []
                        detection_source = "Negative / Blank (No T Line)"
            else:
                detection_source = "Failed Detection"
                C_area, T_area = 1.0, 0.0

            tc_ratio = (T_area / C_area) if C_area > 0 else 0.0

            anchor_idx = final_peaks[0] if len(final_peaks) > 0 else int(np.argmax(green_avg))
            r_val = float(red_avg[anchor_idx])
            g_val = float(green_avg[anchor_idx])
            b_val = float(blue_avg[anchor_idx])
            diff_val = float(g_val - r_val)

            r_peaks.append(r_val)
            g_peaks.append(g_val)
            b_peaks.append(b_val)
            diff_peaks.append(diff_val)

            # Save individual strip report
            with open(os.path.join(conc_dir, f"{file_name}_results.txt"), "w") as tf:
                tf.write(f"File Name: {file_name}\n")
                tf.write(f"Concentration: {concentration}\n")
                tf.write(f"Replicate: {replicate_number}\n")
                tf.write(f"Detection Engine: Two-Stage Cascaded YOLOv8 ({detection_source})\n")
                tf.write(f"Control (C) Line Confidence: {c_conf_val*100:.1f}%\n")
                tf.write(f"Test (T) Line Confidence: {t_conf_val*100:.1f}%\n")
                tf.write("-" * 45 + "\n")
                tf.write(f"Control (C) Area: {C_area:.4f}\n")
                tf.write(f"Test (T) Area: {T_area:.4f}\n")
                tf.write(f"T/C Ratio: {tc_ratio:.6f}\n")
                tf.write("-" * 45 + "\n")
                tf.write("RAW OPTICAL ABSORBANCE AT BAND PEAK:\n")
                tf.write(f"Green Channel (530 nm Absorbance): {g_val:.2f} a.u.\n")
                tf.write(f"Red Channel   (650 nm Reflection): {r_val:.2f} a.u.\n")
                tf.write(f"Blue Channel  (450 nm Absorbance): {b_val:.2f} a.u.\n")
                tf.write(f"Differential Signal (Green - Red): {diff_val:.2f} a.u.\n")

            # Dual-panel publication chromatogram
            fig = plt.figure(figsize=(9.5, 4.8), dpi=300)
            gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 5.0], height_ratios=[4.0, 1.2],
                                   wspace=0.18, hspace=0.10)

            ax1 = fig.add_subplot(gs[0, 0])
            ax_leg = fig.add_subplot(gs[1, 0])
            ax2 = fig.add_subplot(gs[:, 1])

            ax1.imshow(img)
            clean_conc_text = re.sub(r'(?i)\s*ng(/ml)?', '', str(concentration).strip())
            title_disp = f"{clean_conc_text} ng/mL" + (f"\n({replicate_number})" if replicate_number else "")
            ax1.set_title(title_disp, fontsize=9.5, weight='bold', pad=6)
            ax1.axis('off')

            strip_w = img.shape[1]
            if x_c_end > x_c_begin and C_area > 0.0:
                rect_c = patches.Rectangle(
                    (0.5, x_c_begin), strip_w - 1, x_c_end - x_c_begin,
                    linewidth=1.0, edgecolor='#762a83', facecolor='none', linestyle='-'
                )
                ax1.add_patch(rect_c)

            if x_t_end > x_t_begin and T_area > 0.0:
                rect_t = patches.Rectangle(
                    (0.5, x_t_begin), strip_w - 1, x_t_end - x_t_begin,
                    linewidth=1.0, edgecolor='#2166ac', facecolor='none', linestyle='-'
                )
                ax1.add_patch(rect_t)

            line_signal, = ax2.plot(clean_channel, color='#1b7837', linewidth=1.8, label='AuNP Signal (G - R)')
            ax2.set_xlabel('Migration Distance (pixels)', fontsize=10, fontweight='semibold', labelpad=6)
            ax2.set_ylabel('Inverted Absorbance (a.u.)', fontsize=10, fontweight='semibold', labelpad=6)
            ax2.tick_params(direction='out', length=4, width=1, labelsize=9.5)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['left'].set_linewidth(1.0)
            ax2.spines['bottom'].set_linewidth(1.0)

            y_max_profile = max(10, np.max(clean_channel) * 1.30)
            ax2.set_ylim(-0.5, y_max_profile)
            ax2.set_xlim(0, len(clean_channel))

            legend_handles = [line_signal]

            if len(final_peaks) > 0 and len(x_c_baseline) > 0 and C_area > 0.0:
                c_peak_x = final_peaks[0]
                c_peak_y = clean_channel[c_peak_x]
                ax2.plot(x_c_baseline, np.zeros_like(x_c_baseline), color='#762a83', linestyle='--', linewidth=1.2)
                fill_c = ax2.fill_between(x_c_baseline, clean_channel[x_c_begin:x_c_end + 1],
                                         0, color='#af8dc3', alpha=0.55,
                                         label=f'C Line ({C_area:.1f})')
                legend_handles.append(fill_c)
                ax2.annotate(
                    'Control Line',
                    xy=(c_peak_x, c_peak_y),
                    xytext=(c_peak_x, c_peak_y + (y_max_profile * 0.08)),
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#40004b',
                    arrowprops=dict(arrowstyle='->', color='#762a83', lw=1.0, shrinkA=2, shrinkB=2)
                )

            if len(final_peaks) > 1 and len(x_t_baseline) > 0 and T_area > 0.0:
                t_peak_x = final_peaks[1]
                t_peak_y = clean_channel[t_peak_x]
                ax2.plot(x_t_baseline, np.zeros_like(x_t_baseline), color='#2166ac', linestyle='--', linewidth=1.2)
                fill_t = ax2.fill_between(x_t_baseline, clean_channel[x_t_begin:x_t_end + 1],
                                         0, color='#92c5de', alpha=0.55,
                                         label=f'T Line ({T_area:.1f})')
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
            ax2.text(0.97, 0.95, stats_text, transform=ax2.transAxes,
                     fontsize=9.5, fontweight='bold', verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round,pad=0.35', facecolor='#f7f7f7', edgecolor='#cccccc', alpha=0.9))

            plt.savefig(os.path.join(conc_dir, f"{file_name}_Analysis_Plot.png"), bbox_inches='tight', dpi=300)
            plt.close(fig)

            file_names.append(file_name)
            concentrations.append(concentration)
            replicate_nums.append(replicate_number)
            t_areas.append(T_area)
            c_areas.append(C_area)
            tc_ratios.append(tc_ratio)
            c_confs.append(c_conf_val)
            t_confs.append(t_conf_val)
            det_sources.append(detection_source)

        print(f"\rProcessing: Completed!{' ' * 20}\n")

        # ---------------------------------------------------------
        # STATISTICAL ANALYSIS & 4PL CALIBRATION
        # ---------------------------------------------------------
        df = pd.DataFrame({
            'File Name': file_names,
            'Concentration (ng/mL)': concentrations,
            'Replicate': replicate_nums,
            'T': t_areas,
            'C': c_areas,
            'T/C': tc_ratios,
            'C_Conf (%)': [round(c * 100, 1) for c in c_confs],
            'T_Conf (%)': [round(t * 100, 1) for t in t_confs],
            'Detector': det_sources,
            'Green_Abs': g_peaks,
            'Red_Abs': r_peaks,
            'Blue_Abs': b_peaks,
            'Diff_Signal': diff_peaks
        })

        df['Concentration Numeric'] = df['Concentration (ng/mL)'].apply(clean_concentration)
        df.dropna(subset=['Concentration Numeric'], inplace=True)

        grouped = df.groupby('Concentration Numeric')['T/C'].agg(['mean', 'std']).reset_index()
        grouped.rename(columns={'mean': 'Mean T/C', 'std': 'Standard Deviation T/C'}, inplace=True)
        grouped['CV (%)'] = (grouped['Standard Deviation T/C'] / grouped['Mean T/C']) * 100
        grouped.fillna(0, inplace=True)

        print(f"Stats for {folder_name}:\n{grouped.to_string(index=True)}")

        conc_vals = grouped['Concentration Numeric'].values
        avg_tc = grouped['Mean T/C'].values
        std_tc = grouped['Standard Deviation T/C'].values

        hook_text, lod_text, fit_text = "", "", ""
        std_curve_path = None

        if len(avg_tc) > 3:
            conc_fit = conc_vals.copy()
            if len(conc_fit) > 0 and conc_fit[0] == 0:
                non_zero_min = np.min(conc_fit[conc_fit > 0]) if np.any(conc_fit > 0) else 0.01
                conc_fit[0] = non_zero_min / 10.0

            fit_mask, hook_conc, hook_sig, is_hook, hook_text = detect_hook_and_saturation(conc_vals, avg_tc)

            try:
                popt, r2, fit_text = fit_weighted_4pl(conc_fit, avg_tc, fit_mask)
                lob_s, lod_s, lod_c, lod_text, valid_lod = calculate_clsi_limits(avg_tc, std_tc, conc_fit, popt)

                fig_c, ax = plt.subplots(figsize=(6.5, 5.2), dpi=300)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.errorbar(conc_vals, avg_tc, yerr=std_tc, fmt='o', color='#0f2a4a',
                            ecolor='#4a5568', elinewidth=1.2, capsize=3.5,
                            label="Experimental Data (Mean ± SD)", zorder=3)

                x_smooth = np.logspace(np.log10(conc_fit.min() * 0.5), np.log10(conc_fit.max() * 2.0), 350)
                ax.plot(x_smooth, logistic4(x_smooth, *popt), color='#dc2626',
                        linewidth=1.8, label=f"4PL Fit ($R^2$ = {r2:.4f})", zorder=2)

                if valid_lod:
                    ax.plot(lod_c, lod_s, marker='D', markersize=6.5, color='#111827',
                            label=f"LoD: {lod_c:.4f} ng/mL", zorder=4)
                    ax.hlines(lod_s, xmin=conc_fit.min() * 0.5, xmax=lod_c, color='#6b7280', linestyle='--', linewidth=1.0)
                    ax.vlines(lod_c, ymin=0, ymax=lod_s, color='#6b7280', linestyle='--', linewidth=1.0)

                ax.axvspan(hook_conc, conc_fit.max() * 2.5, color='#fef3c7', alpha=0.45, label="Hook / Saturation Zone")
                ax.set_xscale('log')
                ax.set_xlabel("Target Concentration (ng/mL)", fontweight='bold')
                ax.set_ylabel("Signal Ratio (T/C)", fontweight='bold')
                ax.set_title(f"{substrate_name} Standard Curve\n{folder_name}", fontweight='bold')
                ax.set_xlim(conc_fit.min() * 0.5, conc_fit.max() * 2.0)
                ax.set_ylim(0, max(np.max(avg_tc) + np.max(std_tc), popt[3]) * 1.15)
                ax.legend(frameon=True, fontsize=8, loc='lower right')

                std_curve_path = os.path.join(results_dir, f"{folder_name}_Standard_Curve_Plot.png")
                plt.savefig(std_curve_path, bbox_inches='tight', dpi=300)
                plt.close(fig_c)
            except Exception as e:
                fit_text = f"Fit failed: {e}"

        # Subfolder Summary Report
        summary_path = os.path.join(results_dir, f"{folder_name}_Summary_Results.txt")
        with open(summary_path, "w") as sf:
            sf.write("=" * 75 + f"\nANALYSIS SUMMARY: {folder_name}\n" + "=" * 75 + "\n\n")
            sf.write("1. STATISTICAL SUMMARY ACROSS CONCENTRATIONS:\n" + "-" * 75 + f"\n{grouped.to_string(index=True)}\n\n")
            sf.write("2. ANALYTICAL LIMITS & CALIBRATION:\n" + "-" * 75 + f"\n * {hook_text}\n * {lod_text}\n * {fit_text}\n\n")
            sf.write("3. INDIVIDUAL REPLICATES:\n" + "-" * 75 + f"\n{df[['File Name', 'Concentration (ng/mL)', 'Replicate', 'C', 'T', 'T/C', 'Diff_Signal']].to_string(index=False)}\n")

        print(f"Summary saved: {summary_path}")

        execution_summary[folder_name] = {
            'results_dir': results_dir,
            'summary_file': summary_path,
            'standard_curve': std_curve_path,
            'lod_text': lod_text,
            'fit_text': fit_text,
            'grouped_stats': grouped,
            'df': df
        }

    # Optional continuous fine-tuning
    if collected_samples and dataset_dir and os.path.exists(dataset_dir):
        user_choice = input("\nFine-tune YOLO model with newly processed data? (y/n): ").strip().lower()
        if user_choice in ['y', 'yes']:
            fine_tune_model(dataset_dir, strip_weights, collected_samples)

    return execution_summary

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        base_dir=args.input_dir,
        dataset_dir=args.dataset_dir,
        strip_weights=args.strip_weights,
        band_weights=args.band_weights,
        strip_conf=args.strip_conf,
        band_conf=args.band_conf
    )
