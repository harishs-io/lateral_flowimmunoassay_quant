import numpy as np
from scipy.optimize import curve_fit, fsolve
import scipy.stats as stats

def logistic4(x, a, b, c, d):
    """Four-Parameter Logistic (4PL) regression model."""
    return d + (a - d) / (1.0 + (np.maximum(x, 1e-9) / np.maximum(c, 1e-9))**b)

def logistic5(x, a, b, c, d, g):
    """Five-Parameter Logistic (5PL) asymmetric regression model."""
    denom = 1.0 + (np.maximum(x, 1e-9) / np.maximum(c, 1e-9))**b
    return d + (a - d) / (np.maximum(denom, 1e-9)**g)

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
    lower_bounds = [0.0, 0.1, fit_x.min() * 0.05, np.min(fit_y)]
    upper_bounds = [np.max(fit_y), 8.0, fit_x.max() * 5.0, np.max(fit_y) * 2.5]

    weights = 1.0 / np.maximum(fit_y, 0.05)**2
    sigma_weights = 1.0 / np.sqrt(weights)

    popt, _ = curve_fit(
        logistic4, fit_x, fit_y, p0=p0, bounds=(lower_bounds, upper_bounds),
        sigma=sigma_weights, method='trf', maxfev=25000
    )
    a, b, c, d = popt

    residuals = fit_y - logistic4(fit_x, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((fit_y - np.mean(fit_y))**2)
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    fit_text = f"4PL Parameters: a={a:.4f}, b={b:.4f}, c={c:.4f}, d={d:.4f} | R^2 = {r_squared:.4f}"

    return popt, r_squared, ss_res, fit_text

def fit_weighted_5pl(conc_fit, avg_tc, fit_mask):
    """
    Fits a 5PL asymmetric model using 1/y^2 inverse-variance weights.
    """
    fit_x = conc_fit[fit_mask]
    fit_y = avg_tc[fit_mask]

    p0 = [np.min(fit_y), 1.0, np.median(fit_x), np.max(fit_y), 1.0]
    lower_bounds = [0.0, 0.1, fit_x.min() * 0.05, np.min(fit_y), 0.1]
    upper_bounds = [np.max(fit_y), 8.0, fit_x.max() * 5.0, np.max(fit_y) * 2.5, 5.0]

    weights = 1.0 / np.maximum(fit_y, 0.05)**2
    sigma_weights = 1.0 / np.sqrt(weights)

    try:
        popt, _ = curve_fit(
            logistic5, fit_x, fit_y, p0=p0, bounds=(lower_bounds, upper_bounds),
            sigma=sigma_weights, method='trf', maxfev=35000
        )
        a, b, c, d, g = popt

        residuals = fit_y - logistic5(fit_x, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((fit_y - np.mean(fit_y))**2)
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        fit_text = f"5PL Parameters (Asymmetric): a={a:.4f}, b={b:.4f}, c={c:.4f}, d={d:.4f}, g={g:.4f} | R^2 = {r_squared:.4f}"
        return popt, r_squared, ss_res, fit_text, True
    except Exception as e:
        popt_4, r2_4, ss_4, text_4 = fit_weighted_4pl(conc_fit, avg_tc, fit_mask)
        popt = np.append(popt_4, 1.0)
        return popt, r2_4, ss_4, f"5PL fit defaulted to 4PL: {text_4}", False

def adaptive_model_selection(conc_fit, avg_tc, fit_mask):
    """
    Evaluates both 4PL and 5PL models using an Extra Sum-of-Squares F-Test and AIC.
    Selects 5PL only when the reduction in residual variance is statistically significant (p < 0.05).
    """
    n_pts = int(np.sum(fit_mask))
    popt_4pl, r2_4pl, ss_4pl, text_4pl = fit_weighted_4pl(conc_fit, avg_tc, fit_mask)

    if n_pts <= 5:
        # Insufficient degrees of freedom to justify 5 parameters
        decision_info = {
            'model': '4PL',
            'popt': popt_4pl,
            'r_squared': r2_4pl,
            'fit_text': text_4pl,
            'f_stat': None,
            'p_value': None,
            'rationale': f'4PL adopted: Insufficient degrees of freedom (N={n_pts} <= 5) for 5PL.'
        }
        return decision_info

    popt_5pl, r2_5pl, ss_5pl, text_5pl, success_5pl = fit_weighted_5pl(conc_fit, avg_tc, fit_mask)

    df_4pl = n_pts - 4
    df_5pl = n_pts - 5

    # Extra Sum of Squares F-test
    numerator = (ss_4pl - ss_5pl) / max(1, (df_4pl - df_5pl))
    denominator = ss_5pl / max(1, df_5pl)
    f_stat = max(0.0, numerator / denominator) if denominator > 0 else 0.0
    p_val = stats.f.sf(f_stat, 1, df_5pl)

    # Akaike Information Criterion
    aic_4pl = n_pts * np.log(max(1e-9, ss_4pl / n_pts)) + 2 * 4
    aic_5pl = n_pts * np.log(max(1e-9, ss_5pl / n_pts)) + 2 * 5

    if success_5pl and p_val < 0.05 and aic_5pl < aic_4pl and r2_5pl >= r2_4pl:
        chosen_model = '5PL'
        chosen_popt = popt_5pl
        chosen_r2 = r2_5pl
        chosen_text = text_5pl
        rationale = f"5PL adopted: Significant asymmetric plateau fit (F={f_stat:.2f}, p={p_val:.4f} < 0.05, AIC={aic_5pl:.1f} vs {aic_4pl:.1f})."
    else:
        chosen_model = '4PL'
        chosen_popt = popt_4pl
        chosen_r2 = r2_4pl
        chosen_text = text_4pl
        rationale = f"4PL adopted: Parsimonious symmetric fit validated (p={p_val:.4f} >= 0.05, R^2={r2_4pl:.4f})."

    return {
        'model': chosen_model,
        'popt': chosen_popt,
        'r_squared': chosen_r2,
        'fit_text': chosen_text,
        'f_stat': f_stat,
        'p_value': p_val,
        'aic_4pl': aic_4pl,
        'aic_5pl': aic_5pl,
        'ss_4pl': ss_4pl,
        'ss_5pl': ss_5pl,
        'rationale': rationale
    }

def calculate_clsi_limits(avg_tc, std_tc, conc_fit, popt, model_choice='4PL'):
    """
    Computes CLSI EP17-compliant Limit of Blank (LoB) and Limit of Detection (LoD).
    """
    mu_blank = avg_tc[0]
    sigma_blank = std_tc[0] if std_tc[0] > 0 else 0.01
    sigma_low = std_tc[1] if len(std_tc) > 1 and std_tc[1] > 0 else sigma_blank

    lob_signal = mu_blank + 1.645 * sigma_blank
    lod_signal = lob_signal + 1.645 * sigma_low

    def solve_lod(x):
        if model_choice == '5PL':
            return logistic5(x, *popt) - lod_signal
        return logistic4(x, *popt[:4]) - lod_signal

    lod_conc = fsolve(solve_lod, x0=conc_fit.min())[0]

    c_param = popt[2]
    if conc_fit.min() * 0.1 <= lod_conc <= c_param:
        lod_text = f"Calculated LoD = {lod_conc:.4f} ng/mL (LoD Signal = {lod_signal:.4f}, LoB Signal = {lob_signal:.4f})"
        valid = True
    else:
        lod_text = f"Analytical LoD Signal ({lod_signal:.4f}) falls outside dynamic binding range."
        valid = False

    return lob_signal, lod_signal, lod_conc, lod_text, valid

class InversionResult(tuple):
    """
    Subclass of tuple (conc, status) maintaining backward-compatible 2-tuple unpacking
    while exposing analytical confidence intervals, classification zone, and dilution factor.
    """
    def __new__(cls, conc, status, ci_low=None, ci_high=None, zone="Quantified", dilution_factor=1.0):
        return super().__new__(cls, (conc, status))

    def __init__(self, conc, status, ci_low=None, ci_high=None, zone="Quantified", dilution_factor=1.0):
        self.conc = conc
        self.status = status
        self.ci_low = ci_low
        self.ci_high = ci_high
        self.zone = zone
        self.dilution_factor = dilution_factor

def invert_concentration(y_tc, popt, model_choice='4PL', lod_signal=None, dilution_factor=1.0, 
                         se_reg=None, df=None, n_pts=None, x_mean=None, ss_x=None):
    """
    Inverts the calibrated 4PL or 5PL equation to compute analyte concentration from optical T/C.
    Handles boundaries gracefully and computes 95% Student's t Prediction Intervals via Delta Method.
    Returns InversionResult which unrolls as (conc, status) or accesses .ci_low, .ci_high, .zone.
    """
    a = float(popt[0])
    b = float(popt[1])
    c = float(popt[2])
    d = float(popt[3])
    df_val = dilution_factor if dilution_factor > 0 else 1.0

    # Below LoD or Detection Floor Check
    if (lod_signal is not None and y_tc < lod_signal) or y_tc <= a:
        status = f"Below Detection Limit (< LoD signal {lod_signal:.4f})" if lod_signal else "Below Analytical Baseline (< LoD)"
        return InversionResult(0.0, status, ci_low=0.0, ci_high=0.0, zone="Below LoD", dilution_factor=df_val)

    # Upper Plateau Saturation Check
    if y_tc >= d:
        status = f"Upper Saturation Plateau (> ULoQ signal {d:.4f})"
        return InversionResult(float('inf'), status, ci_low=None, ci_high=None, zone="Saturated (> ULoQ)", dilution_factor=df_val)

    try:
        if model_choice == '5PL':
            g = float(popt[4]) if len(popt) > 4 else 1.0
            term = ((a - d) / (y_tc - d))**(1.0 / g) - 1.0
        else:
            term = (a - d) / (y_tc - d) - 1.0

        if term <= 0:
            return InversionResult(float('inf'), "Upper Saturation Plateau (> ULoQ)", 
                                   ci_low=None, ci_high=None, zone="Saturated (> ULoQ)", dilution_factor=df_val)

        conc_neat = c * (term**(1.0 / b))
        conc_final = max(0.0, float(conc_neat * df_val))

        # Calculate 95% Student's t Prediction Interval via Delta Method
        ci_low, ci_high = None, None
        if se_reg is not None and df is not None and df > 0:
            x_c = max(1e-9, conc_neat / max(c, 1e-9))
            if model_choice == '5PL':
                g = float(popt[4]) if len(popt) > 4 else 1.0
                denom = 1.0 + x_c**b
                dy_dx = (a - d) * (-g) * (denom**(-g - 1.0)) * (b * (x_c**(b - 1.0)) * (1.0 / c))
            else:
                denom = (1.0 + x_c**b)**2
                dy_dx = -(a - d) * b * (x_c**(b - 1.0)) * (1.0 / c) / denom

            slope = abs(dy_dx)
            n_val = max(1, n_pts if n_pts is not None else (df + 4))
            xm_val = x_mean if x_mean is not None else conc_neat
            ssx_val = ss_x if ss_x is not None else 100.0
            leverage = 1.0 + (1.0 / n_val) + ((conc_neat - xm_val)**2 / max(1e-4, ssx_val))
            sx = (se_reg / max(1e-6, slope)) * np.sqrt(leverage)
            t_crit = stats.t.ppf(0.975, df)
            delta_x = t_crit * sx * df_val
            ci_low = max(0.0, float(conc_final - delta_x))
            ci_high = float(conc_final + delta_x)
            status = f"Quantified in Dynamic Range [{ci_low:.4f} – {ci_high:.4f}]"
        else:
            status = "Quantified in Dynamic Range"

        return InversionResult(conc_final, status, ci_low=ci_low, ci_high=ci_high, 
                               zone="Quantified", dilution_factor=df_val)

    except Exception as e:
        return InversionResult(None, f"Mathematical inversion error: {str(e)}", 
                               ci_low=None, ci_high=None, zone="Error", dilution_factor=df_val)

def verify_hook_dilution(y_neat=None, y_diluted=None, dilution_factor=10.0, popt=None, model_choice='4PL',
                         lod_signal=None, se_reg=None, df=None, neat_tc=None, diluted_tc=None):
    """
    Two-Point Dilution Parallelism / Hook Disruption Test (CLSI EP34 / FDA BMV).
    Compares neat vs 1:10 diluted sample strips to resolve Hook effect vs True Low concentration.
    Supports both y_neat/y_diluted and neat_tc/diluted_tc parameter names.
    """
    if neat_tc is not None:
        y_neat = neat_tc
    if diluted_tc is not None:
        y_diluted = diluted_tc

    if y_neat is None or y_diluted is None:
        raise ValueError("Both neat and diluted T/C measurements are required.")

    delta_y = float(y_diluted - y_neat)

    # If diluted strip has a stronger optical signal than neat (Δy > 0.05), free antigen hook block was relieved!
    if delta_y > 0.05:
        is_hook = True
        status = "Hook Effect Confirmed: Free antigen blockade relieved upon dilution (Δy > 0.05)."
        if popt is not None:
            res_dil = invert_concentration(y_diluted, popt, model_choice=model_choice,
                                           lod_signal=lod_signal, dilution_factor=dilution_factor,
                                           se_reg=se_reg, df=df)
            conc_final = res_dil.conc
            ci_low = res_dil.ci_low
            ci_high = res_dil.ci_high
        else:
            conc_final, ci_low, ci_high = None, None, None
    else:
        is_hook = False
        status = "True Low Concentration Confirmed: Signal remained low upon dilution (Δy <= 0.05)."
        if popt is not None:
            res_neat = invert_concentration(y_neat, popt, model_choice=model_choice,
                                            lod_signal=lod_signal, dilution_factor=1.0,
                                            se_reg=se_reg, df=df)
            conc_final = res_neat.conc
            ci_low = res_neat.ci_low
            ci_high = res_neat.ci_high
        else:
            conc_final, ci_low, ci_high = None, None, None

    return {
        'is_hook': is_hook,
        'delta_y': delta_y,
        'status': status,
        'final_concentration': conc_final,
        'ci_low': ci_low,
        'ci_high': ci_high
    }
