import numpy as np
from scipy.signal import find_peaks, peak_widths

integrate_area = getattr(np, "trapezoid", getattr(np, "trapz", None))

def extract_lspr_signal(img, smooth=False, top_cutoff_ratio=0.15, bottom_cutoff_ratio=0.85):
    """
    Extracts the differential colloidal gold localized surface plasmon resonance (LSPR)
    profile: Inverted Green (530 nm absorption) - Inverted Red (650 nm reflection).
    """
    red_channel = img[:, :, 0]
    green_channel = img[:, :, 1]
    blue_channel = img[:, :, 2]

    red_average = 255.0 - np.mean(red_channel, axis=1)
    green_average = 255.0 - np.mean(green_channel, axis=1)
    blue_average = 255.0 - np.mean(blue_channel, axis=1)

    diff_signal = green_average - red_average
    best_channel = np.clip(diff_signal, 0, 255)
    best_channel[best_channel < 5] = 0

    if smooth:
        window_size = 5
        best_channel = np.convolve(best_channel, np.ones(window_size)/window_size, mode='same')

    signal_len = len(best_channel)
    best_channel[:int(signal_len * top_cutoff_ratio)] = 0
    best_channel[int(signal_len * bottom_cutoff_ratio):] = 0

    return best_channel, red_average, green_average, blue_average

def identify_bands(best_channel, min_dist=40, min_prom=3.0, shadow_max=11.0, 
                   conj_pad_cutoff=500, min_gap=45, max_gap=90):
    """
    Detects and filters peaks to isolate Control and Test bands using 
    shadow suppression, cutoff filtering, and spatial pair matching.
    """
    raw_peaks, properties = find_peaks(
        best_channel, distance=min_dist, prominence=min_prom, width=1
    )
    sort_order = np.argsort(raw_peaks)
    sorted_peaks = list(raw_peaks[sort_order])
    sorted_proms = list(properties['prominences'][sort_order])

    # Filter absorbent pad shadow artifact
    if len(sorted_peaks) > 0 and best_channel[sorted_peaks[0]] <= shadow_max:
        sorted_peaks.pop(0)
        sorted_proms.pop(0)

    # Filter conjugate pad border
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
    """
    Integrates Control (C) and Test (T) band areas using local sloped dynamic baselines.
    """
    c_area, t_area = 0.0, 0.0
    geom_data = {
        'x_c_base': [], 'y_c_base': [], 'c_begin': 0, 'c_end': 0,
        'x_t_base': [], 'y_t_base': [], 't_begin': 0, 't_end': 0
    }

    if len(final_peaks) > 0:
        widths_half, _, _, _ = peak_widths(best_channel, final_peaks, rel_height=0.5)

        # Control Line Integration
        c_idx = final_peaks[0]
        w_c = widths_half[0]
        x_c_begin = max(0, int(round(c_idx - 2 * w_c)))
        x_c_end = min(len(best_channel) - 1, int(round(c_idx + 2 * w_c)))
        y_c_begin = best_channel[x_c_begin]
        y_c_end = best_channel[x_c_end]

        slope_c = (y_c_end - y_c_begin) / (x_c_end - x_c_begin) if (x_c_end - x_c_begin) != 0 else 0
        x_c_base = np.arange(x_c_begin, x_c_end + 1)
        y_c_base = y_c_end + (x_c_base - x_c_end) * slope_c

        for i, x in enumerate(x_c_base):
            if best_channel[x] < y_c_base[i]:
                best_channel[x] = y_c_base[i]

        c_area = (
            integrate_area(best_channel[x_c_begin:x_c_end + 1], x_c_base)
            - integrate_area([y_c_begin, y_c_end], [x_c_begin, x_c_end])
        )
        geom_data.update({'x_c_base': x_c_base, 'y_c_base': y_c_base, 'c_begin': x_c_begin, 'c_end': x_c_end})

        # Test Line Integration
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

            for i, x in enumerate(x_t_base):
                if best_channel[x] < y_t_base[i]:
                    best_channel[x] = y_t_base[i]

            t_area = (
                integrate_area(best_channel[x_t_begin:x_t_end + 1], x_t_base)
                - integrate_area([y_t_begin, y_t_end], [x_t_begin, x_t_end])
            )
            geom_data.update({'x_t_base': x_t_base, 'y_t_base': y_t_base, 't_begin': x_t_begin, 't_end': x_t_end})

    tc_ratio = t_area / c_area if c_area != 0 else 0.0
    return c_area, t_area, tc_ratio, geom_data
