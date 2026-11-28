import pandas as pd


def _metric_degraded(metric):
    if metric["value"] is None or metric["previous_value"] is None:
        return False
    higher_is_better = metric["name"] in ("accuracy", "precision", "recall", "f1_score", "roc_auc", "r2")
    return metric["value"] < metric["previous_value"] if higher_is_better else metric["value"] > metric["previous_value"]


def monitoring_agent(metrics, health_status):
    available = [m for m in metrics if m["value"] is not None]
    changes = [m["name"] for m in available if _metric_degraded(m)]
    return {"summary": f"Health is {health_status}.",
            "degraded_metrics": changes,
            "available_metrics": [m["name"] for m in available]}


def drift_agent(drift_features):
    drifted = [f for f in drift_features if f["drift_detected"]]
    return {"summary": f"{len(drifted)} of {len(drift_features)} selected features show drift.",
            "features": [{**f, "explanation": f["explanation"]} for f in drifted]}


def root_cause_agent(metrics, drift_features):
    degraded = [m["name"] for m in metrics if _metric_degraded(m)]
    suspicious = [f["feature"] for f in drift_features
                  if f["severity"] in ("Warning", "Critical")]
    return {"possible_causes": (
        [f"Degraded metrics may coincide with drift in: {', '.join(suspicious)}"] if degraded and suspicious
        else ["No evidence-based link can be established from the available data."]
    ), "confidence": "correlation only; no causal claim"}


def recommendation_agent(health_status, metrics, drift_features):
    recommendations = []
    if health_status == "Insufficient Data":
        return {"recommendations": [
            {"priority": 1, "action": "Provide valid numeric or categorical feature values and production ground-truth labels to assess model performance."},
            {"priority": 2, "action": "Confirm the selected features match the trained model's expected inputs."},
        ]}
    severe = [f["feature"] for f in drift_features if f["severity"] == "Critical"]
    if severe:
        recommendations.append({"priority": 1, "action": f"Investigate critical drift in {', '.join(severe)}."})
    degraded = [m["name"] for m in metrics if _metric_degraded(m)]
    if degraded:
        recommendations.append({"priority": 2, "action": "Evaluate retraining with recent production data."})
    recommendations.append({"priority": 3, "action": "Collect additional labeled production data before retraining."})
    if health_status == "Healthy":
        recommendations = [{"priority": 4, "action": "Continue routine monitoring."}]
    return {"recommendations": recommendations}
