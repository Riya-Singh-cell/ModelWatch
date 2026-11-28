from io import BytesIO

import joblib
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression, LogisticRegression

from app.main import app

client = TestClient(app)


def _upload_model():
    training = pd.DataFrame({"signal": [0, 1, 2, 3, 7, 8, 9, 10]})
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    estimator = LogisticRegression().fit(training, labels)
    artifact = BytesIO()
    joblib.dump(estimator, artifact)
    artifact.seek(0)
    response = client.post(
        "/api/model-workspaces",
        data={"trusted_source": "true"},
        files={"file": ("trained.pkl", artifact.getvalue(), "application/octet-stream")},
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


def _upload_regression_model():
    training = pd.DataFrame({"hours": [1, 2, 3, 4, 5, 6], "activity_Yes": [0, 1, 0, 1, 0, 1]})
    estimator = LinearRegression().fit(training, [10, 20, 30, 40, 50, 60])
    artifact = BytesIO()
    joblib.dump(estimator, artifact)
    response = client.post(
        "/api/model-workspaces",
        data={"trusted_source": "true"},
        files={"file": ("regressor.pkl", artifact.getvalue(), "application/octet-stream")},
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


def _upload_csv(model_id, dataset_type, csv_data):
    return client.post(
        f"/api/models/{model_id}/datasets?dataset_type={dataset_type}",
        files={"file": (f"{dataset_type}.csv", csv_data, "text/csv")},
    )


def test_model_workspace_runs_actual_predictions_and_persists_results():
    model_id = _upload_model()
    reference = "signal,label\n0,0\n1,0\n2,0\n8,1\n9,1\n"
    production = "signal,label\n1,0\n2,0\n8,1\n9,1\n"
    try:
        assert client.get(f"/api/models/{model_id}").json()["status"] == "Not monitored"
        assert client.get("/api/dashboard").json()["overall_status"] == "Not monitored"
        original_upload = _upload_csv(model_id, "reference", reference)
        current_upload = _upload_csv(model_id, "production", production)
        assert original_upload.status_code == 200
        assert original_upload.json()["row_count"] == 5
        assert original_upload.json()["column_count"] == 2
        assert len(original_upload.json()["preview"]) == 5
        assert current_upload.json()["compatibility"]["shared_columns"] == ["label", "signal"]

        run = client.post(f"/api/models/{model_id}/monitor", json={
            "independent_features": ["signal"],
            "target_feature": "label",
        })
        assert run.status_code == 200, run.text
        result = run.json()
        metrics = {item["name"]: item["value"] for item in result["metrics"]}
        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1_score"] == 1.0
        assert metrics["roc_auc"] is not None
        assert len(result["agents"]) == 5
        assert result["agents"]["LLM Analysis"]["status"] == "unconfigured"
        assert client.get(f"/api/models/{model_id}/runs").json()[0]["id"] == result["id"]
        details = client.get(f"/api/models/{model_id}/runs").json()[0]
        assert details["status_factors"]["performance_metrics_available"] is True
        assert details["status_factors"]["performance_metric_count"] == 5
    finally:
        client.delete(f"/api/models/{model_id}")


def test_regression_model_calculates_regression_metrics_and_encodes_categories():
    model_id = _upload_regression_model()
    try:
        assert client.get(f"/api/models/{model_id}").json()["model_type"] == "regression"
        _upload_csv(model_id, "reference",
                    "hours,activity,target\n1,No,10\n2,Yes,20\n3,No,30\n4,Yes,40\n")
        _upload_csv(model_id, "production",
                    "hours,activity,target\n5,No,50\n6,Yes,60\n")
        result = client.post(f"/api/models/{model_id}/monitor", json={
            "independent_features": ["hours", "activity"],
            "target_feature": "target",
        })
        assert result.status_code == 200, result.text
        payload = result.json()
        assert payload["model_type"] == "regression"
        assert {metric["name"] for metric in payload["metrics"]} == {"mae", "rmse", "r2"}
        assert all(metric["value"] is not None for metric in payload["metrics"])
        assert payload["status_factors"]["performance_metric_count"] == 3
    finally:
        client.delete(f"/api/models/{model_id}")


def test_missing_production_target_marks_supervised_metrics_unavailable():
    model_id = _upload_model()
    try:
        assert _upload_csv(model_id, "reference", "signal,label\n0,0\n9,1\n").status_code == 200
        assert _upload_csv(model_id, "production", "signal\n1\n8\n").status_code == 200
        run = client.post(f"/api/models/{model_id}/monitor", json={
            "independent_features": ["signal"],
            "target_feature": "label",
        })
        assert run.status_code == 200, run.text
        assert all(metric["value"] is None for metric in run.json()["metrics"])
    finally:
        client.delete(f"/api/models/{model_id}")


def test_model_upload_and_feature_validation_reject_invalid_inputs():
    unsupported = client.post(
        "/api/model-workspaces",
        data={"trusted_source": "true"},
        files={"file": ("model.txt", b"not a model", "text/plain")},
    )
    untrusted = client.post(
        "/api/model-workspaces",
        data={"trusted_source": "false"},
        files={"file": ("model.pkl", b"not a model", "application/octet-stream")},
    )
    assert unsupported.status_code == 422
    assert untrusted.status_code == 422

    model_id = _upload_model()
    try:
        _upload_csv(model_id, "reference", "signal,label\n0,0\n9,1\n")
        _upload_csv(model_id, "production", "signal,label\n1,0\n8,1\n")
        invalid = client.post(f"/api/models/{model_id}/monitor", json={
            "independent_features": ["not_a_column"],
            "target_feature": "signal",
        })
        assert invalid.status_code == 422
    finally:
        client.delete(f"/api/models/{model_id}")
