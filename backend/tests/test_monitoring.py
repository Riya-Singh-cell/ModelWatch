import pandas as pd
from app.monitoring.metrics import calculate_metrics
from app.monitoring.metrics import calculate_regression_metrics
from app.monitoring.drift import psi, calculate_feature_drift
from app.services.health import evaluate_health
from app.services.validation import read_csv_upload


def test_calculate_metrics():
    result = calculate_metrics([0, 0, 1, 1], [0, 1, 1, 1], [0.1, 0.4, 0.8, 0.9])
    assert result["accuracy"] == 0.75
    assert result["precision"] == 2 / 3
    assert result["recall"] == 1.0
    assert 0.0 <= result["roc_auc"] <= 1.0


def test_missing_ground_truth_metrics_are_unavailable():
    result = calculate_metrics([], [])
    assert all(value is None for value in result.values())


def test_psi_is_zero_for_identical_distributions():
    values = pd.Series([1, 2, 3, 4, 5])
    assert psi(values, values) == 0.0


def test_drift_detected_for_shifted_feature():
    reference = pd.DataFrame({"age": [20, 21, 22, 23, 24, 25]})
    production = pd.DataFrame({"age": [50, 51, 52, 53, 54, 55]})
    result = calculate_feature_drift(reference, production, ["age"])
    assert result[0]["drift_detected"] is True
    assert result[0]["ks_p_value"] < 0.05


def test_health_evaluation():
    metrics = [{"name": "accuracy", "value": 0.70, "previous_value": 0.90}]
    status, explanation = evaluate_health(metrics, [])
    assert status == "Warning"
    assert "accuracy" in explanation


def test_health_evaluation_uses_correct_direction_for_regression_errors():
    metrics = [{"name": "rmse", "value": 12.0, "previous_value": 8.0}]
    status, explanation = evaluate_health(metrics, [])
    assert status == "Warning"
    assert "rmse" in explanation


def test_health_is_not_claimed_without_any_monitoring_evidence():
    status, explanation = evaluate_health([], [])
    assert status == "Insufficient Data"
    assert "no supervised metrics" in explanation


def test_regression_metrics_are_real_and_directionally_interpretable():
    result = calculate_regression_metrics([1, 2, 3], [1, 3, 2])
    assert result["mae"] == 2 / 3
    assert result["rmse"] > 0
    assert result["r2"] < 1


def test_csv_validation_rejects_empty_data():
    try:
        read_csv_upload("bad.csv", b"")
    except ValueError as exc:
        assert "empty" in str(exc).lower()
    else:
        raise AssertionError("Empty CSV should be rejected")
