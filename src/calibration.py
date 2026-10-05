import numpy as np
from scipy.optimize import curve_fit, fsolve

def logistic4(x, a, b, c, d):
    """Four-Parameter Logistic (4PL) regression model."""
    return d + (a - d) / (1.0 + (x / np.maximum(c, 1e-9))**b)

def detect_hook_and_saturation(conc_values, avg_tc):
    """
    Identifies high-dose hook inhibition (prozone effect) or receptor site saturation plateaus.
    Returns dynamic fitting masks restricting regression to the physiological binding range.
    """
    max_idx = np.argmax(avg_tc)
    hook_conc = conc_values[max_idx]
    hook_signal = avg_tc[max_idx]

    is_hook = False
    if max_idx < len(avg_tc) - 1 and avg_tc[max_idx] > avg_tc[-1] * 1.05:
        is_hook = True
        fit_mask = np.arange(len(conc_values)) <= (max_idx + 1)
        status_text = f"High-Dose Hook Effect detected at > {hook_conc:.2f} ng/mL (Peak T/C = {hook_signal:.3f})"
    else:
        fit_mask = np.ones(len(conc_values), dtype=bool)
        status_text = f"Membrane Saturation Plateau reached at >= {hook_conc:.2f} ng/mL (Max T/C = {hook_signal:.3f})"

    return fit_mask, hook_conc, hook_signal, is_hook, status_text

def fit_weighted_4pl(conc_fit, avg_tc, fit_mask):
    """
    Fits a 4PL model using 1/y^2 inverse-variance weights to anchor calibration near LoD.
    """
    fit_x = conc_fit[fit_mask]
    fit_y = avg_tc[fit_mask]

    p0 = [np.min(fit_y), 1.0, np.median(fit_x), np.max(fit_y)]
    lower_bounds = [0.0, 0.1, fit_x.min() * 0.1, np.min(fit_y)]
    upper_bounds = [np.max(fit_y), 5.0, fit_x.max() * 5.0, np.max(fit_y) * 2.0]

    weights = 1.0 / np.maximum(fit_y, 0.05)**2
    sigma_weights = 1.0 / np.sqrt(weights)

    popt, _ = curve_fit(
        logistic4, fit_x, fit_y, p0=p0, bounds=(lower_bounds, upper_bounds),
        sigma=sigma_weights, method='trf', maxfev=20000
    )
    a, b, c, d = popt

    residuals = fit_y - logistic4(fit_x, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((fit_y - np.mean(fit_y))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    fit_text = f"4PL Parameters (Dynamic Fit): a={a:.4f}, b={b:.4f}, c={c:.4f}, d={d:.4f} | R^2 = {r_squared:.4f}"

    return popt, r_squared, fit_text

def calculate_clsi_limits(avg_tc, std_tc, conc_fit, popt):
    """
    Computes CLSI EP17-compliant Limit of Blank (LoB) and Limit of Detection (LoD).
    """
    a, b, c, d = popt
    mu_blank = avg_tc[0]
    sigma_blank = std_tc[0] if std_tc[0] > 0 else 0.01
    sigma_low = std_tc[1] if len(std_tc) > 1 and std_tc[1] > 0 else sigma_blank

    lob_signal = mu_blank + 1.645 * sigma_blank
    lod_signal = lob_signal + 1.645 * sigma_low

    def solve_lod(x):
        return logistic4(x, a, b, c, d) - lod_signal

    lod_conc = fsolve(solve_lod, x0=conc_fit.min())[0]

    if conc_fit.min() * 0.1 <= lod_conc <= c:
        lod_text = f"Calculated LoD = {lod_conc:.4f} ng/mL (LoD Signal = {lod_signal:.4f}, LoB Signal = {lob_signal:.4f})"
        valid = True
    else:
        lod_text = f"Analytical LoD Signal ({lod_signal:.4f}) falls outside dynamic binding range."
        valid = False

    return lob_signal, lod_signal, lod_conc, lod_text, valid
