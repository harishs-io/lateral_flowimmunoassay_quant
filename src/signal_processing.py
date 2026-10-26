import numpy as np
from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import grey_opening

integrate_area = getattr(np, "trapezoid", getattr(np, "trapz", None))

MIN_PEAK_PROMINENCE = 5.0    # Minimum net optical peak height (a.u.) for a real band
MIN_SADDLE_DELTA = 2.5       # Minimum height rise above local baseline floor

def extract_lspr_signal(img, smooth=False, top_cutoff_ratio=0.10, bottom_cutoff_ratio=0.90):
    """
    Extracts the differential colloidal gold localized surface plasmon resonance (LSPR)
    profile: Inverted Green (530 nm absorption) - Inverted Red (650 nm reflection).
    Equivalent to: I_Red - I_Green.
    """
    red_channel = img[:, :, 0]
    green_channel = img[:, :, 1]
    blue_channel = img[:, :, 2]

    red_mean_raw = np.mean(red_channel, axis=1)
    green_mean_raw = np.mean(green_channel, axis=1)
    blue_mean_raw = np.mean(blue_channel, axis=1)

    red_average = 255.0 - red_mean_raw
    green_average = 255.0 - green_mean_raw
    blue_average = 255.0 - blue_mean_raw

    color_specific_signal = green_average - red_average
    raw_best_channel = np.clip(color_specific_signal, 0, 255)

    if smooth:
        window_size = 5
        raw_best_channel = np.convolve(raw_best_channel, np.ones(window_size)/window_size, mode='same')

    signal_len = len(raw_best_channel)
    raw_best_channel[:int(signal_len * top_cutoff_ratio)] = 0
    raw_best_channel[int(signal_len * bottom_cutoff_ratio):] = 0

    return raw_best_channel, red_average, green_average, blue_average, red_mean_raw, green_mean_raw, blue_mean_raw

def extract_longitudinal_baseline(raw_sig, filter_radius=35):
    """
    Computes a full-strip longitudinal continuum baseline envelope using
    morphological grey opening to eliminate membrane unevenness and vignetting.
    """
    struct_size = max(5, int(filter_radius * 2 + 1))
    baseline_env = grey_opening(raw_sig, size=struct_size)
    baseline_env = np.minimum(baseline_env, raw_sig)
    return baseline_env

def integrate_band_adaptive(raw_sig, baseline_env, y_center, expected_width=16,
                            membrane_height=1000.0, rel_threshold=0.05,
                            min_prominence=MIN_PEAK_PROMINENCE, min_saddle=MIN_SADDLE_DELTA):
    """
    Integrates an LFIA band around y_center using 5% apex boundary thresholding,
    strict noise-floor prominence gating, and Rf-normalized AUC integration.
    """
    sig_len = len(raw_sig)
    yc = int(np.clip(round(y_center), 0, sig_len - 1))
    half_search = max(4, int(round(expected_width * 1.5)))
    w_start = max(0, yc - half_search)
    w_end = min(sig_len - 1, yc + half_search)

    empty_result = (0, 0, 0, np.array([]), np.array([]), 0.0, 0.0, 0.0, 0.0, 0.0)
    if w_end <= w_start + 1:
        return empty_result

    local_segment = raw_sig[w_start:w_end + 1]
    local_base = baseline_env[w_start:w_end + 1]
    net_local = np.maximum(0, local_segment - local_base)

    apex_offset = int(np.argmax(net_local))
    apex = w_start + apex_offset
    apex_val = float(net_local[apex_offset])

    # Peak validation against local membrane background noise
    left_min = float(np.min(net_local[:apex_offset])) if apex_offset > 0 else float(net_local[0])
    right_min = float(np.min(net_local[apex_offset:])) if apex_offset < len(net_local) - 1 else float(net_local[-1])
    saddle = max(left_min, right_min)
    prominence = apex_val - saddle

    # Reject noise ripples or non-peaks on blank strips
    if apex_val < min_prominence or prominence < min_saddle:
        return empty_result

    # 5% Apex boundary cutoff
    cutoff_val = apex_val * rel_threshold
    l_idx = apex_offset
    while l_idx > 0 and net_local[l_idx - 1] > cutoff_val:
        l_idx -= 1

    r_idx = apex_offset
    while r_idx < len(net_local) - 1 and net_local[r_idx + 1] > cutoff_val:
        r_idx += 1

    start = w_start + l_idx
    end = w_start + r_idx
    if end <= start:
        end = min(sig_len - 1, start + 1)

    xs = np.arange(start, end + 1)
    seg = raw_sig[start:end + 1]

    flat_base_val = float(np.min(baseline_env[max(0, start - 5):min(sig_len, end + 6)]))
    flat_baseline = np.full_like(seg, flat_base_val)
    net_peak = np.maximum(0, seg - flat_baseline)

    rf_dx = 1000.0 / float(max(1, membrane_height))

    net_area = float(integrate_area(net_peak, xs) * (rf_dx if len(xs) > 1 else 0.0))
    raw_area = float(integrate_area(seg, xs) * (rf_dx if len(xs) > 1 else 0.0))
    base_area = float(integrate_area(flat_baseline, xs) * (rf_dx if len(xs) > 1 else 0.0))
    raw_pk = float(np.max(seg))
    net_pk = float(np.max(net_peak))

    return apex, start, end, xs, net_peak, net_area, raw_area, base_area, raw_pk, net_pk

# Backward compatibility functions
def identify_bands(best_channel, min_dist=40, min_prom=3.0, shadow_max=11.0,
                   conj_pad_cutoff=500, min_gap=45, max_gap=90):
    """Legacy 1D topological band detector (used as secondary fallback)."""
    raw_peaks, properties = find_peaks(
        best_channel, distance=min_dist, prominence=min_prom, width=1
    )
    if len(raw_peaks) == 0:
        return np.array([])
    sort_order = np.argsort(raw_peaks)
    sorted_peaks = list(raw_peaks[sort_order])
    sorted_proms = list(properties['prominences'][sort_order])

    if len(sorted_peaks) > 0 and best_channel[sorted_peaks[0]] <= shadow_max:
        sorted_peaks.pop(0)
        sorted_proms.pop(0)

    candidate_peaks = [p for p in sorted_peaks if p <= conj_pad_cutoff]
    candidate_proms = [sorted_proms[i] for i, p in enumerate(sorted_peaks) if p <= conj_pad_cutoff]

    final_peaks = []
    if len(candidate_peaks) > 0:
        valid_pairs = []
        for i in range(len(candidate_peaks)):
            for j in range(i + 1, len(candidate_peaks)):
                if min_gap <= (candidate_peaks[j] - candidate_peaks[i]) <= max_gap:
                    valid_pairs.append((candidate_peaks[i], candidate_peaks[j]))

        if len(valid_pairs) > 0:
            best_pair = max(valid_pairs, key=lambda pair: best_channel[pair[0]] + best_channel[pair[1]])
            final_peaks = [best_pair[0], best_pair[1]]
        else:
            strongest_idx = np.argmax(candidate_proms)
            final_peaks = [candidate_peaks[strongest_idx]]

    return np.array(final_peaks)

def integrate_peaks(best_channel, final_peaks):
    """Legacy 1D trapezoidal peak integrator (used as secondary fallback)."""
    c_area, t_area = 0.0, 0.0
    geom_data = {
        'x_c_base': [], 'y_c_base': [], 'c_begin': 0, 'c_end': 0,
        'x_t_base': [], 'y_t_base': [], 't_begin': 0, 't_end': 0
    }

    if len(final_peaks) > 0:
        widths_half, _, _, _ = peak_widths(best_channel, final_peaks, rel_height=0.5)
        c_idx = final_peaks[0]
        w_c = widths_half[0]
        x_c_begin = max(0, int(round(c_idx - 2 * w_c)))
        x_c_end = min(len(best_channel) - 1, int(round(c_idx + 2 * w_c)))
        y_c_begin = best_channel[x_c_begin]
        y_c_end = best_channel[x_c_end]

        slope_c = (y_c_end - y_c_begin) / (x_c_end - x_c_begin) if (x_c_end - x_c_begin) != 0 else 0
        x_c_base = np.arange(x_c_begin, x_c_end + 1)
        y_c_base = y_c_end + (x_c_base - x_c_end) * slope_c

        c_area = (
            integrate_area(best_channel[x_c_begin:x_c_end + 1], x_c_base)
            - integrate_area([y_c_begin, y_c_end], [x_c_begin, x_c_end])
        )
        geom_data.update({'x_c_base': x_c_base, 'y_c_base': y_c_base, 'c_begin': x_c_begin, 'c_end': x_c_end})

        if len(final_peaks) > 1:
            t_idx = final_peaks[1]
            w_t = widths_half[1]
            x_t_begin = max(0, int(round(t_idx - 2 * w_t)))
            x_t_end = min(len(best_channel) - 1, int(round(t_idx + 2 * w_t)))
            y_t_begin = best_channel[x_t_begin]
            y_t_end = best_channel[x_t_end]

            slope_t = (y_t_end - y_t_begin) / (x_t_end - x_t_begin) if (x_t_end - x_t_begin) != 0 else 0
            x_t_base = np.arange(x_t_begin, x_t_end + 1)
            y_t_base = y_t_end + (x_t_base - x_t_end) * slope_t

            t_area = (
                integrate_area(best_channel[x_t_begin:x_t_end + 1], x_t_base)
                - integrate_area([y_t_begin, y_t_end], [x_t_begin, x_t_end])
            )
            geom_data.update({'x_t_base': x_t_base, 'y_t_base': y_t_base, 't_begin': x_t_begin, 't_end': x_t_end})

    tc_ratio = t_area / c_area if c_area != 0 else 0.0
    return c_area, t_area, tc_ratio, geom_data
