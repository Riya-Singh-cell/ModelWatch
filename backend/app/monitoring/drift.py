import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def psi(expected, actual, bins=10):
    expected = pd.to_numeric(pd.Series(expected), errors="coerce").dropna()
    actual = pd.to_numeric(pd.Series(actual), errors="coerce").dropna()
    if expected.empty or actual.empty:
        return None
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(expected.quantile(quantiles).to_numpy())
    if len(edges) < 2:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    exp_counts = np.histogram(expected, bins=edges)[0].astype(float)
    act_counts = np.histogram(actual, bins=edges)[0].astype(float)
    exp_dist = np.where(exp_counts == 0, 1e-6, exp_counts / exp_counts.sum())
    act_dist = np.where(act_counts == 0, 1e-6, act_counts / act_counts.sum())
    return float(np.sum((act_dist - exp_dist) * np.log(act_dist / exp_dist)))


def _categorical_psi(expected, actual):
    expected = pd.Series(expected).dropna().astype(str)
    actual = pd.Series(actual).dropna().astype(str)
    if expected.empty or actual.empty:
        return None
    categories = sorted(set(expected) | set(actual))
    expected_counts = expected.value_counts().reindex(categories, fill_value=0).to_numpy(dtype=float)
    actual_counts = actual.value_counts().reindex(categories, fill_value=0).to_numpy(dtype=float)
    expected_dist = np.maximum(expected_counts / expected_counts.sum(), 1e-6)
    actual_dist = np.maximum(actual_counts / actual_counts.sum(), 1e-6)
    return float(np.sum((actual_dist - expected_dist) * np.log(actual_dist / expected_dist)))


def calculate_feature_drift(reference: pd.DataFrame, production: pd.DataFrame,
                            columns, ks_alpha=0.05, psi_thresholds=None):
    psi_thresholds = psi_thresholds or {"warning": 0.1, "critical": 0.25}
    results = []
    for column in columns:
        if column not in reference or column not in production:
            continue
        ref_source = reference[column].dropna()
        prod_source = production[column].dropna()
        ref = pd.to_numeric(ref_source, errors="coerce").dropna()
        prod = pd.to_numeric(prod_source, errors="coerce").dropna()
        categorical = (
            (not ref_source.empty and len(ref) < len(ref_source))
            or (not prod_source.empty and len(prod) < len(prod_source))
        )
        if categorical:
            psi_value = _categorical_psi(ref_source, prod_source)
            severity = "None"
            if psi_value is not None:
                if psi_value >= psi_thresholds["critical"]:
                    severity = "Critical"
                elif psi_value >= psi_thresholds["warning"]:
                    severity = "Warning"
            drift_detected = severity in ("Warning", "Critical")
            results.append({
                "feature": column,
                "ks_statistic": None,
                "ks_p_value": None,
                "psi": psi_value,
                "drift_detected": drift_detected,
                "severity": severity if psi_value is not None else "Unavailable",
                "explanation": (
                    f"{column}: categorical feature; KS test is unavailable. "
                    f"PSI is {psi_value:.4f} ({severity.lower()} severity)."
                    if psi_value is not None else f"{column}: insufficient observations for PSI."
                ),
            })
            continue
        if ref.empty or prod.empty:
            results.append({"feature": column, "ks_statistic": None, "ks_p_value": None,
                            "psi": None, "drift_detected": False, "severity": "Unavailable",
                            "explanation": "Insufficient numeric observations."})
            continue
        ks = ks_2samp(ref, prod)
        psi_value = psi(ref, prod)
        ks_drift = bool(ks.pvalue < ks_alpha)
        severity = "None"
        if psi_value is not None:
            if psi_value >= psi_thresholds["critical"]:
                severity = "Critical"
            elif psi_value >= psi_thresholds["warning"]:
                severity = "Warning"
        drift_detected = ks_drift or severity in ("Warning", "Critical")
        explanation = (
            f"{column}: KS p-value {ks.pvalue:.4f} "
            f"{'indicates' if ks_drift else 'does not indicate'} significant distribution change; "
            f"PSI is {psi_value:.4f} ({severity.lower() if psi_value is not None else 'unavailable'} severity)."
        )
        results.append({"feature": column, "ks_statistic": float(ks.statistic),
                        "ks_p_value": float(ks.pvalue), "psi": psi_value,
                        "drift_detected": drift_detected, "severity": severity,
                        "explanation": explanation})
    return results
