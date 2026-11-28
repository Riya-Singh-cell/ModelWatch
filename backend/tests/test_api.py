from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_register_model_and_negative_missing_model():
    payload = {
        "name": "API Test Model",
        "model_type": "classification",
        "description": "test",
        "target_column": "label",
        "feature_columns": ["age", "income"],
    }
    created = client.post("/api/models", json=payload)
    assert created.status_code == 201
    model_id = created.json()["id"]
    assert client.get(f"/api/models/{model_id}").status_code == 200
    assert client.get("/api/models/999999").status_code == 404


def test_upload_and_monitor_pipeline():
    model = client.post("/api/models", json={
        "name": "Pipeline Model", "model_type": "classification",
        "description": "", "target_column": "label",
        "feature_columns": ["age", "income"],
    }).json()
    model_id = model["id"]
    reference = ("age,income,label\n20,30000,0\n21,32000,0\n22,34000,1\n23,36000,1\n")
    production = ("age,income,label,_prediction,_probability\n50,80000,0,0,0.2\n"
                  "51,82000,1,1,0.8\n52,84000,1,1,0.9\n53,86000,1,1,0.7\n")
    ref = client.post(f"/api/models/{model_id}/datasets?dataset_type=reference",
                      files={"file": ("reference.csv", reference, "text/csv")})
    prod = client.post(f"/api/models/{model_id}/datasets?dataset_type=production",
                       files={"file": ("production.csv", production, "text/csv")})
    assert ref.status_code == 200
    assert prod.status_code == 200
    run = client.post(f"/api/models/{model_id}/monitor")
    assert run.status_code == 200
    body = run.json()
    assert body["health_status"] in ("Healthy", "Warning", "Critical")
    assert len(body["metrics"]) == 5
    assert len(body["drift"]["features"]) == 2
    assert len(body["agents"]) == 5
    assert body["agents"]["LLM Analysis"]["status"] == "unconfigured"
