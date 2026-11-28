import json

import httpx
import pytest

from app.agents import llm


def _alternative_recommendations(evidence_id="health.status", candidate=None):
    models = [candidate] if candidate else []
    return {
        "replacement_required": bool(models),
        "reason": "No alternative is justified by the current evidence." if not models else (
            "The supplied performance metric supports evaluating a compatible alternative."
        ),
        "recommended_models": models,
        "next_steps": ["Compare candidates on the same labeled validation data."],
        "evidence_ids": [evidence_id],
    }


def _structured_response(evidence_id="metric.accuracy", alternatives=None):
    return {
        "summary": "Performance is stable in the supplied evidence.",
        "findings": [{
            "severity": "info",
            "title": "No material regression",
            "explanation": "The current accuracy is consistent with the available reference run.",
            "evidence_ids": [evidence_id],
        }],
        "possible_causes": [],
        "recommendations": [{
            "priority": 2,
            "action": "Continue scheduled monitoring.",
            "rationale": "Current evidence shows no significant performance regression.",
            "evidence_ids": [evidence_id],
        }],
        "alternative_model_recommendations": alternatives or _alternative_recommendations(),
        "limitations": ["This analysis does not establish causal relationships."],
    }


def _run_analysis(
    metrics,
    drift_features,
    health_status,
    health_explanation,
    problem_type="classification",
    algorithm="LogisticRegression",
    features=None,
    performance_degradations=None,
):
    return llm.analyze_monitoring_results(
        metrics,
        drift_features,
        health_status,
        health_explanation,
        model_context={
            "name": "Test workspace",
            "algorithm": algorithm,
            "problem_type": problem_type,
            "features": features or ["signal"],
            "fit_diagnostics_available": False,
        },
        performance_degradations=performance_degradations or [],
    )


def test_llm_status_does_not_expose_secret(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "do-not-return-this")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    status = llm.llm_status()
    assert status == {
        "configured": True,
        "provider": "OpenAI-compatible",
        "model": "test-model",
        "message": None,
    }
    assert "do-not-return-this" not in str(status)


def test_unconfigured_llm_returns_explicit_status(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    result = _run_analysis([], [], "Healthy", "All good.")
    assert result["status"] == "unconfigured"
    assert result["structured_output"] is None
    assert "LLM_API_KEY" in result["message"]


def test_llm_response_is_schema_validated_and_only_aggregate_evidence_is_sent(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-secret")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["payload"] = kwargs["json"]
        response_body = {
            "choices": [{
                "message": {"content": json.dumps(_structured_response())}
            }]
        }
        return httpx.Response(200, json=response_body, request=httpx.Request("POST", url))

    monkeypatch.setattr(llm.httpx, "post", fake_post)
    result = _run_analysis(
        [{"name": "accuracy", "value": 0.9, "previous_value": 0.89}],
        [],
        "Healthy",
        "No significant drift.",
        algorithm="LogisticRegression",
        features=["signal", "age"],
    )
    assert result["status"] == "complete"
    assert result["structured_output"]["findings"][0]["evidence_ids"] == ["metric.accuracy"]
    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    assert captured["payload"]["response_format"]["type"] == "json_schema"
    request_content = captured["payload"]["messages"][1]["content"]
    assert "0.9" in request_content
    assert "raw_dataset_row" not in request_content
    request_data = json.loads(request_content)
    assert request_data["current_model"]["algorithm"] == "LogisticRegression"
    assert request_data["current_model"]["features"] == ["signal", "age"]
    assert request_data["problem_type"] == "classification"


def test_unknown_evidence_reference_is_rejected(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-secret")
    monkeypatch.setenv("LLM_MODEL", "test-model")

    def fake_post(url, **_kwargs):
        return httpx.Response(200, json={
            "choices": [{
                "message": {"content": json.dumps(_structured_response("invented.evidence"))}
            }]
        }, request=httpx.Request("POST", url))

    monkeypatch.setattr(llm.httpx, "post", fake_post)
    result = _run_analysis(
        [{"name": "accuracy", "value": 0.9, "previous_value": 0.89}],
        [],
        "Healthy",
        "No significant drift.",
    )
    assert result["status"] == "error"
    assert result["structured_output"] is None


def test_llm_provider_failure_is_reported_without_hiding_deterministic_results(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-secret")
    monkeypatch.setenv("LLM_MODEL", "test-model")

    def fake_post(url, **_kwargs):
        return httpx.Response(
            503,
            json={"error": {"message": "provider temporarily unavailable"}},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(llm.httpx, "post", fake_post)
    result = _run_analysis(
        [{"name": "accuracy", "value": 0.9, "previous_value": 0.89}],
        [],
        "Healthy",
        "No significant drift.",
    )
    assert result["status"] == "error"
    assert result["structured_output"] is None
    assert "deterministic monitoring results are still available" in result["message"].lower()


@pytest.mark.parametrize(
    ("problem_type", "algorithm", "metric", "value", "previous_value", "candidate_name"),
    [
        ("regression", "LinearRegression", "r2", -4.19, 0.8, "Random Forest Regressor"),
        ("classification", "LogisticRegression", "accuracy", 0.4, 0.9, "Random Forest Classifier"),
    ],
)
def test_alternative_recommendations_match_task_and_cite_performance(
    monkeypatch, problem_type, algorithm, metric, value, previous_value, candidate_name
):
    monkeypatch.setenv("LLM_API_KEY", "test-secret")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    evidence_id = f"metric.{metric}"
    candidate = {
        "model_name": candidate_name,
        "priority": 1,
        "reason": "A tree ensemble may capture nonlinear patterns.",
        "expected_advantage": "Potentially improves fit if the relationships are nonlinear.",
        "evaluation_required": True,
        "evidence_ids": [evidence_id],
    }
    response_body = {
        "choices": [{
            "message": {"content": json.dumps(_structured_response(
                evidence_id=evidence_id,
                alternatives=_alternative_recommendations(evidence_id, candidate),
            ))}
        }]
    }
    captured = {}

    def fake_post(url, **kwargs):
        captured["payload"] = kwargs["json"]
        return httpx.Response(200, json=response_body, request=httpx.Request("POST", url))

    monkeypatch.setattr(llm.httpx, "post", fake_post)
    result = _run_analysis(
        [{"name": metric, "value": value, "previous_value": previous_value}],
        [],
        "Warning",
        "A performance metric is available.",
        problem_type=problem_type,
        algorithm=algorithm,
        performance_degradations=[metric],
    )

    assert result["status"] == "complete"
    recommendation = result["structured_output"]["alternative_model_recommendations"]
    assert recommendation["replacement_required"] is True
    assert recommendation["recommended_models"][0]["model_name"] == candidate_name
    prompt = json.loads(captured["payload"]["messages"][1]["content"])
    assert candidate_name in prompt["allowed_alternative_algorithms"]
    assert prompt["problem_type"] == problem_type
    assert prompt["performance_degradation_detected"] is True


def test_drift_without_labels_never_recommends_an_alternative(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-secret")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    drift_only_candidate = {
        "model_name": "Random Forest Regressor",
        "priority": 1,
        "reason": "The input data has drifted.",
        "expected_advantage": "A different model may adapt.",
        "evaluation_required": True,
        "evidence_ids": ["drift.signal"],
    }

    def fake_post(url, **_kwargs):
        return httpx.Response(200, json={
            "choices": [{
                "message": {"content": json.dumps(_structured_response(
                    evidence_id="health.status",
                    alternatives=_alternative_recommendations("health.status", drift_only_candidate),
                ))}
            }]
        }, request=httpx.Request("POST", url))

    monkeypatch.setattr(llm.httpx, "post", fake_post)
    result = _run_analysis(
        [],
        [{
            "feature": "signal",
            "ks_statistic": 0.9,
            "ks_p_value": 0.001,
            "psi": 0.3,
            "drift_detected": True,
            "severity": "Critical",
        }],
        "Warning",
        "Critical feature drift; no ground-truth labels.",
        problem_type="regression",
        algorithm="LinearRegression",
    )

    assert result["status"] == "complete"
    recommendation = result["structured_output"]["alternative_model_recommendations"]
    assert recommendation["replacement_required"] is False
    assert recommendation["recommended_models"] == []
    assert "drift alone" in recommendation["reason"].lower()


def test_cross_problem_type_alternative_is_rejected(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-secret")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    incompatible_candidate = {
        "model_name": "Random Forest Classifier",
        "priority": 1,
        "reason": "A classification model could be useful.",
        "expected_advantage": "May fit the data better.",
        "evaluation_required": True,
        "evidence_ids": ["metric.r2"],
    }

    def fake_post(url, **_kwargs):
        return httpx.Response(200, json={
            "choices": [{
                "message": {"content": json.dumps(_structured_response(
                    evidence_id="metric.r2",
                    alternatives=_alternative_recommendations("metric.r2", incompatible_candidate),
                ))}
            }]
        }, request=httpx.Request("POST", url))

    monkeypatch.setattr(llm.httpx, "post", fake_post)
    result = _run_analysis(
        [{"name": "r2", "value": -1.0, "previous_value": None}],
        [],
        "Warning",
        "Poor regression metric.",
        problem_type="regression",
        algorithm="LinearRegression",
    )

    assert result["status"] == "error"
    assert result["structured_output"] is None

