def find_performance_degradations(metrics, threshold=0.05):
    degradations = []
    for metric in metrics:
        if metric["value"] is None or metric["previous_value"] is None:
            continue
        change = (metric["value"] - metric["previous_value"]) / max(abs(metric["previous_value"]), 1e-9)
        lower_is_worse = metric["name"] in ("accuracy", "precision", "recall", "f1_score", "roc_auc", "r2")
        worsened = change < -threshold if lower_is_worse else change > threshold
        if worsened:
            degradations.append(metric["name"])
    return degradations


def evaluate_health(metrics, drift_features, config=None):
    config = config or {"metric_degradation": 0.05, "drift_warning": 1, "drift_critical": 3}
    degradations = find_performance_degradations(metrics, config["metric_degradation"])
    critical_drift = sum(1 for item in drift_features if item["severity"] == "Critical")
    warning_drift = sum(1 for item in drift_features if item["severity"] == "Warning")
    has_metric_evidence = any(metric["value"] is not None for metric in metrics)
    has_drift_evidence = any(item["severity"] != "Unavailable" for item in drift_features)
    if critical_drift >= config["drift_critical"] or len(degradations) >= 3:
        status = "Critical"
        signals = []
        if critical_drift >= config["drift_critical"]:
            signals.append(
                f"{critical_drift} features reached critical PSI severity "
                f"(threshold for overall Critical: {config['drift_critical']})"
            )
        if len(degradations) >= 3:
            signals.append(f"{len(degradations)} performance metrics degraded beyond the configured threshold")
        explanation = "Critical: " + "; ".join(signals) + ". Investigate the listed signals before relying on current predictions."
    elif critical_drift or warning_drift or degradations:
        status = "Warning"
        signals = []
        if critical_drift:
            signals.append(
                f"{critical_drift} feature(s) have critical PSI drift "
                f"(overall Critical requires {config['drift_critical']})"
            )
        if warning_drift:
            signals.append(f"{warning_drift} feature(s) have warning-level PSI drift")
        if degradations:
            signals.append(f"performance worsened for {', '.join(degradations)}")
        explanation = "Warning: " + "; ".join(signals) + ". Review drifted features and compare recent labeled performance."
    elif not has_metric_evidence and not has_drift_evidence:
        status = "Insufficient Data"
        explanation = "Insufficient data: no supervised metrics or usable feature-drift statistics were available. Check target labels and numeric/categorical feature values."
    elif not has_metric_evidence:
        status = "Healthy"
        explanation = "No significant feature drift was detected. Supervised performance could not be verified because production ground-truth labels were unavailable."
    else:
        status = "Healthy"
        explanation = "Healthy: supervised performance is available and no substantial metric degradation or significant feature drift was detected."
    return status, explanation
