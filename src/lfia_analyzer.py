import os
import glob
import argparse
import warnings
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO

from src.utils import clean_concentration
from src.detector import auto_crop_strip, fine_tune_model
from src.signal_processing import extract_lspr_signal, identify_bands, integrate_peaks
from src.calibration import logistic4, detect_hook_and_saturation, fit_weighted_4pl, calculate_clsi_limits

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Times', 'serif']
plt.rcParams['mathtext.fontset'] = 'stix'
warnings.filterwarnings("ignore")

def parse_args():
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="LFIA Signal Quantification Pipeline")
    parser.add_argument("--input_dir", type=str, default=str(repo_root / "demo_data"),
                        help="Root folder containing assay subdirectories.")
    parser.add_argument("--weights", type=str, default=str(repo_root / "weights" / "best.pt"),
                        help="Path to YOLOv8 model weights (.pt).")
    parser.add_argument("--dataset_dir", type=str, default=str(repo_root / "dataset"),
                        help="Root directory for continuous learning dataset.")
    parser.add_argument("--conf", type=float, default=0.20, help="YOLO confidence threshold.")
    return parser.parse_args()

def run_pipeline(base_dir, dataset_dir, weights_path, conf_thresh):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found at '{weights_path}'. Check weights/ directory.")

    ml_detector = YOLO(weights_path)
    collected_samples = []

    subfolders = [f.path for f in os.scandir(base_dir) if f.is_dir() and "Results" not in f.path]
    if not subfolders:
        subfolders = [base_dir]

    for folder in subfolders:
        folder_name = os.path.basename(folder)
        print(f"\n{'='*60}\nANALYZING FOLDER: {folder_name}\n{'='*60}")

        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(glob.glob(os.path.join(folder, ext)))

        if not image_files:
            print(f"No LFA images found in '{folder_name}'. Skipping...")
            continue

        image_files.sort(reverse=True)
        file_names, concentrations, replicate_nums = [], [], []
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

            best_channel, red_avg, green_avg, blue_avg = extract_lspr_signal(img)
            final_peaks = identify_bands(best_channel)

            anchor_idx = final_peaks[0] if len(final_peaks) > 0 else np.argmax(green_avg)
            r_val, g_val, b_val = float(red_avg[anchor_idx]), float(green_avg[anchor_idx]), float(blue_avg[anchor_idx])
            diff_val = float(g_val - r_val)
            r_peaks.append(r_val); g_peaks.append(g_val); b_peaks.append(b_val); diff_peaks.append(diff_val)

            c_area, t_area, tc_ratio, geom = integrate_peaks(best_channel, final_peaks)

            # Save individual strip report
            with open(os.path.join(conc_dir, f"{file_name}_results.txt"), "w") as tf:
                tf.write(f"File Name: {file_name}\nConcentration: {concentration}\nReplicate: {replicate_number}\n")
                tf.write("-" * 35 + f"\nControl (C) Area: {c_area:.4f}\nTest (T) Area: {t_area:.4f}\nT/C Ratio: {tc_ratio:.6f}\n")
                tf.write("-" * 35 + f"\nGreen Absorbance (530 nm): {g_val:.2f} a.u.\nRed Reflection (650 nm): {r_val:.2f} a.u.\nDifferential Signal (G-R): {diff_val:.2f} a.u.\n")

            # Save individual chromatographic plot
            fig = plt.figure(figsize=(9.5, 4.8), dpi=300)
            gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 5.0], height_ratios=[4.0, 1.2], wspace=0.18, hspace=0.10)
            ax1, ax_leg, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])

            ax1.imshow(img)
            clean_conc = re.sub(r'(?i)\s*ng(/ml)?', '', str(concentration).strip())
            ax1.set_title(f"{clean_conc} ng/mL" + (f"\n({replicate_number})" if replicate_number else ""), fontsize=9.5, weight='bold')
            ax1.axis('off')

            line_signal, = ax2.plot(best_channel, color='#1b7837', linewidth=1.8, label='AuNP Signal (G - R)')
            ax2.set_xlabel('Migration Distance (pixels)', fontsize=10, fontweight='semibold')
            ax2.set_ylabel('Inverted Absorbance (a.u.)', fontsize=10, fontweight='semibold')
            ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
            y_max_profile = max(10, np.max(best_channel) * 1.30)
            ax2.set_ylim(-0.5, y_max_profile); ax2.set_xlim(0, len(best_channel))

            legend_handles = [line_signal]
            if len(final_peaks) > 0:
                ax2.plot(geom['x_c_base'], geom['y_c_base'], color='#762a83', linestyle='--', linewidth=1.2)
                legend_handles.append(ax2.fill_between(geom['x_c_base'], best_channel[geom['c_begin']:geom['c_end']+1], geom['y_c_base'], color='#af8dc3', alpha=0.55, label=f'C Line ({c_area:.1f})'))
            if len(final_peaks) > 1:
                ax2.plot(geom['x_t_base'], geom['y_t_base'], color='#2166ac', linestyle='--', linewidth=1.2)
                legend_handles.append(ax2.fill_between(geom['x_t_base'], best_channel[geom['t_begin']:geom['t_end']+1], geom['y_t_base'], color='#92c5de', alpha=0.55, label=f'T Line ({t_area:.1f})'))

            ax_leg.axis('off')
            ax_leg.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1.1), fontsize=7.5, frameon=True)
            ax2.text(0.97, 0.95, f"T/C Ratio: {tc_ratio:.4f}", transform=ax2.transAxes, fontsize=9.5, fontweight='bold', ha='right', va='top', bbox=dict(boxstyle='round,pad=0.35', facecolor='#f7f7f7', edgecolor='#cccccc'))
            plt.savefig(os.path.join(conc_dir, f"{file_name}_Analysis_Plot.png"), bbox_inches='tight', dpi=300)
            plt.close(fig)

            file_names.append(file_name); concentrations.append(concentration); replicate_nums.append(replicate_number)
            t_areas.append(t_area); c_areas.append(c_area); tc_ratios.append(tc_ratio)

        print(f"\rProcessing: Completed!{' ' * 20}\n")

        # Statistical Aggregation
        df = pd.DataFrame({'File Name': file_names, 'Concentration (ng/mL)': concentrations, 'Replicate': replicate_nums,
                           'T': t_areas, 'C': c_areas, 'T/C': tc_ratios, 'Green_Abs': g_peaks, 'Red_Abs': r_peaks, 'Blue_Abs': b_peaks, 'Diff_Signal': diff_peaks})
        df['Concentration Numeric'] = df['Concentration (ng/mL)'].apply(clean_concentration)
        df.dropna(subset=['Concentration Numeric'], inplace=True)

        grouped = df.groupby('Concentration Numeric')['T/C'].agg(['mean', 'std']).reset_index()
        grouped.rename(columns={'mean': 'Mean T/C', 'std': 'Standard Deviation T/C'}, inplace=True)
        grouped['CV (%)'] = (grouped['Standard Deviation T/C'] / grouped['Mean T/C']) * 100
        grouped.fillna(0, inplace=True)

        print(f"Stats for {folder_name}:\n{grouped.to_string(index=True)}")

        conc_vals, avg_tc, std_tc = grouped['Concentration Numeric'].values, grouped['Mean T/C'].values, grouped['Standard Deviation T/C'].values
        hook_text, lod_text, fit_text = "", "", ""

        if len(avg_tc) > 3:
            conc_fit = conc_vals.copy()
            if len(conc_fit) > 0 and conc_fit[0] == 0:
                conc_fit[0] = (np.min(conc_fit[conc_fit > 0]) / 10.0) if np.any(conc_fit > 0) else 0.01

            fit_mask, hook_conc, hook_sig, is_hook, hook_text = detect_hook_and_saturation(conc_vals, avg_tc)

            try:
                popt, r2, fit_text = fit_weighted_4pl(conc_fit, avg_tc, fit_mask)
                lob_s, lod_s, lod_c, lod_text, valid_lod = calculate_clsi_limits(avg_tc, std_tc, conc_fit, popt)

                fig_c, ax = plt.subplots(figsize=(6.5, 5.2), dpi=300)
                ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
                ax.errorbar(conc_vals, avg_tc, yerr=std_tc, fmt='o', color='#0f2a4a', ecolor='#4a5568', elinewidth=1.2, capsize=3.5, label="Experimental Data (Mean ± SD)", zorder=3)
                x_smooth = np.logspace(np.log10(conc_fit.min() * 0.5), np.log10(conc_fit.max() * 2.0), 350)
                ax.plot(x_smooth, logistic4(x_smooth, *popt), color='#dc2626', linewidth=1.8, label=f"4PL Fit ($R^2$ = {r2:.4f})", zorder=2)

                if valid_lod:
                    ax.plot(lod_c, lod_s, marker='D', markersize=6.5, color='#111827', label=f"LoD: {lod_c:.4f} ng/mL", zorder=4)
                    ax.hlines(lod_s, xmin=conc_fit.min() * 0.5, xmax=lod_c, color='#6b7280', linestyle='--', linewidth=1.0)
                    ax.vlines(lod_c, ymin=0, ymax=lod_s, color='#6b7280', linestyle='--', linewidth=1.0)

                ax.axvspan(hook_conc, conc_fit.max() * 2.5, color='#fef3c7', alpha=0.45, label="Hook / Saturation Zone")
                ax.set_xscale('log'); ax.set_xlabel("Target Concentration (ng/mL)", fontweight='bold')
                ax.set_ylabel("Signal Ratio (T/C)", fontweight='bold')
                ax.set_title(f"{substrate_name} Standard Curve\n{folder_name}", fontweight='bold')
                ax.set_xlim(conc_fit.min() * 0.5, conc_fit.max() * 2.0)
                ax.set_ylim(0, max(np.max(avg_tc) + np.max(std_tc), popt[3]) * 1.15)
                ax.legend(frameon=True, fontsize=8, loc='lower right')
                plt.savefig(os.path.join(results_dir, f"{folder_name}_Standard_Curve_Plot.png"), bbox_inches='tight', dpi=300)
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

    if collected_samples and os.path.exists(dataset_dir):
        user_choice = input("\nFine-tune YOLO model with newly processed data? (y/n): ").strip().lower()
        if user_choice in ['y', 'yes']:
            fine_tune_model(dataset_dir, weights_path, collected_samples)

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.input_dir, args.dataset_dir, args.weights, args.conf)
