import json
import logging
import os
from pathlib import Path
from typing import Literal

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError

logger = logging.getLogger(__name__)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


class LLMFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    severity: Literal["info", "warning", "critical"]
    title: str = Field(max_length=160)
    explanation: str = Field(max_length=1000)
    evidence_ids: list[str] = Field(min_length=1, max_length=8)


class LLMCause(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hypothesis: str = Field(max_length=500)
    evidence_ids: list[str] = Field(min_length=1, max_length=8)
    confidence: Literal["low", "medium", "high"]
    caveat: str = Field(max_length=500)


class LLMRecommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    priority: int = Field(ge=1, le=4)
    action: str = Field(max_length=400)
    rationale: str = Field(max_length=700)
    evidence_ids: list[str] = Field(min_length=1, max_length=8)


class AlternativeModelCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: Literal[
        "Random Forest Regressor",
        "XGBoost Regressor",
        "Gradient Boosting Regressor",
        "Decision Tree Regressor",
        "Ridge Regression",
        "Lasso Regression",
        "Random Forest Classifier",
        "XGBoost Classifier",
        "Gradient Boosting Classifier",
        "Logistic Regression",
        "Support Vector Machine",
        "Decision Tree Classifier",
    ]
    priority: int = Field(ge=1, le=3)
    reason: str = Field(max_length=700)
    expected_advantage: str = Field(max_length=500)
    evaluation_required: bool
    evidence_ids: list[str] = Field(min_length=1, max_length=8)


class AlternativeModelRecommendations(BaseModel):
    model_config = ConfigDict(extra="forbid")

    replacement_required: bool
    reason: str = Field(max_length=1000)
    recommended_models: list[AlternativeModelCandidate] = Field(max_length=3)
    next_steps: list[str] = Field(min_length=1, max_length=6)
    evidence_ids: list[str] = Field(min_length=1, max_length=8)


class StructuredMonitoringAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(max_length=1200)
    findings: list[LLMFinding] = Field(max_length=10)
    possible_causes: list[LLMCause] = Field(max_length=8)
    recommendations: list[LLMRecommendation] = Field(max_length=8)
    alternative_model_recommendations: AlternativeModelRecommendations
    limitations: list[str] = Field(max_length=8)


def llm_status():
    api_key = os.getenv("LLM_API_KEY", "").strip()
    model = os.getenv("LLM_MODEL", "").strip()
    configured = bool(api_key and model)
    return {
        "configured": configured,
        "provider": "OpenAI-compatible",
        "model": model if configured else None,
        "message": None if configured else "Set LLM_API_KEY and LLM_MODEL on the backend to enable structured LLM analysis.",
    }


def _evidence_catalog(metrics, drift_features, health_status, health_explanation):
    evidence = {
        "health.status": {"label": "Deterministic health status", "value": health_status},
        "health.explanation": {"label": "Deterministic health explanation", "value": health_explanation},
    }
    for metric in metrics:
        evidence[f"metric.{metric['name']}"] = {
            "label": f"Current {metric['name']} metric",
            "value": metric["value"],
            "previous_value": metric["previous_value"],
        }
    for feature in drift_features:
        feature_id = feature["feature"]
        evidence[f"drift.{feature_id}"] = {
            "label": f"Drift statistics for feature {feature_id}",
            "ks_statistic": feature["ks_statistic"],
            "ks_p_value": feature["ks_p_value"],
            "psi": feature["psi"],
            "drift_detected": feature["drift_detected"],
            "severity": feature["severity"],
        }
    return evidence


def analyze_monitoring_results(
    metrics,
    drift_features,
    health_status,
    health_explanation,
    model_context,
    performance_degradations,
):
    configuration = llm_status()
    if not configuration["configured"]:
        return {
            "status": "unconfigured",
            "provider": configuration["provider"],
            "model": None,
            "message": configuration["message"],
            "structured_output": None,
        }

    api_key = os.environ["LLM_API_KEY"].strip()
    model = os.environ["LLM_MODEL"].strip()
    base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    evidence = _evidence_catalog(metrics, drift_features, health_status, health_explanation)
    metric_evidence_ids = {
        f"metric.{item['name']}" for item in metrics if item["value"] is not None
    }
    allowed_models = (
        [
            "Random Forest Regressor",
            "XGBoost Regressor",
            "Gradient Boosting Regressor",
            "Decision Tree Regressor",
            "Ridge Regression",
            "Lasso Regression",
        ]
        if model_context["problem_type"] == "regression"
        else [
            "Random Forest Classifier",
            "XGBoost Classifier",
            "Gradient Boosting Classifier",
            "Logistic Regression",
            "Support Vector Machine",
            "Decision Tree Classifier",
        ]
    )
    drifted_features = [
        {
            "feature": feature["feature"],
            "psi": feature["psi"],
            "ks_p_value": feature["ks_p_value"],
            "severity": feature["severity"],
        }
        for feature in drift_features
        if feature["drift_detected"]
    ]
    metric_summary = {
        item["name"]: {"current": item["value"], "previous": item["previous_value"]}
        for item in metrics
    }
    schema = StructuredMonitoringAnalysis.model_json_schema()
    request_payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a careful ML monitoring analyst. Interpret only the supplied aggregate "
                    "evidence. The deterministic health status, metrics and drift tests are authoritative: "
                    "never change or recompute them. Do not claim correlation proves causation. Do not "
                    "invent facts. Every finding, cause and recommendation must cite existing evidence_ids. "
                    "When evidence is insufficient, say so in limitations and avoid speculative causes. "
                    "Alternative algorithms are hypotheses to evaluate, never automatic replacements. "
                    "Never recommend alternatives solely because of feature drift. A candidate must be "
                    "supported by available supervised performance metric evidence and cite at least one "
                    "non-null metric evidence ID. Distinguish metric degradation from drift. If metrics are "
                    "unavailable, recommend no alternatives. If existing-model retraining is sufficient, "
                    "say so. Select only from the provided task-compatible algorithm list, and do not "
                    "recommend algorithms for the other problem type. Do not claim overfitting or "
                    "underfitting because training/validation diagnostics are unavailable."
                ),
            },
            {
                "role": "user",
                "content": json.dumps({
                    "current_model": model_context,
                    "problem_type": model_context["problem_type"],
                    "performance_metrics": metric_summary,
                    "performance_degradation_detected": bool(performance_degradations),
                    "degraded_metrics": performance_degradations,
                    "drifted_features": drifted_features,
                    "fit_diagnostics": {
                        "available": model_context["fit_diagnostics_available"],
                        "details": None if model_context["fit_diagnostics_available"] else (
                            "Training and validation scores were not saved, so overfitting or "
                            "underfitting cannot be assessed."
                        ),
                    },
                    "allowed_alternative_algorithms": allowed_models,
                    "health_status": health_status,
                    "evidence": evidence,
                    "output_contract": (
                        "Return a concise user-facing summary, findings, possible non-causal hypotheses, "
                        "prioritized recommendations, alternative_model_recommendations, and limitations. "
                        "Recommend at most three relevant alternatives. replacement_required means an "
                        "evaluation is justified, not that production should be replaced. If no alternative "
                        "is justified, return false and an empty recommended_models list. For each candidate, "
                        "explain why and the potential advantage without promising improved performance. "
                        "Use only evidence IDs in the evidence map; candidate evidence_ids must include a "
                        "non-null performance metric. Include next_steps for validation using the same "
                        "labeled validation data and metrics. Do not emit unsupported model types."
                    ),
                }),
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "modelwatch_monitoring_analysis",
                "strict": True,
                "schema": schema,
            },
        },
    }

    try:
        response = httpx.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=request_payload,
            timeout=httpx.Timeout(30.0, connect=5.0),
        )
        response.raise_for_status()
        body = response.json()
        content = body["choices"][0]["message"]["content"]
        analysis = StructuredMonitoringAnalysis.model_validate_json(content)
        allowed_evidence_ids = set(evidence)
        alternative = analysis.alternative_model_recommendations
        cited_items = [
            *analysis.findings,
            *analysis.possible_causes,
            *analysis.recommendations,
            alternative,
            *alternative.recommended_models,
        ]
        for item in cited_items:
            invalid_ids = set(item.evidence_ids) - allowed_evidence_ids
            if invalid_ids:
                raise ValueError(f"LLM returned unknown evidence IDs: {', '.join(sorted(invalid_ids))}")
        incompatible_models = {
            candidate.model_name for candidate in alternative.recommended_models
            if candidate.model_name not in allowed_models
        }
        if incompatible_models:
            raise ValueError(
                f"LLM returned models incompatible with {model_context['problem_type']}: "
                f"{', '.join(sorted(incompatible_models))}"
            )
        if bool(alternative.recommended_models) != alternative.replacement_required:
            raise ValueError(
                "replacement_required must match whether recommended_models contains candidates."
            )
        if any(not candidate.evaluation_required for candidate in alternative.recommended_models):
            raise ValueError("Alternative model candidates must require validation before replacement.")
        if not metric_evidence_ids:
            drifted_feature_names = [
                feature["feature"] for feature in drift_features if feature["drift_detected"]
            ]
            next_steps = []
            if drifted_feature_names:
                next_steps.append(
                    f"Investigate observed drift in {', '.join(drifted_feature_names)}."
                )
            else:
                next_steps.append(
                    "Collect labeled production outcomes before assessing predictive performance."
                )
            next_steps.extend([
                "Collect labeled production outcomes and rerun monitoring.",
                "Reassess alternatives only if supervised performance is inadequate.",
            ])
            alternative = AlternativeModelRecommendations(
                replacement_required=False,
                reason=(
                    "Production ground-truth metrics are unavailable, so this run cannot establish "
                    "that another algorithm would perform better. Feature drift alone is not a reason "
                    "to recommend a replacement."
                ),
                recommended_models=[],
                next_steps=next_steps,
                evidence_ids=["health.status", "health.explanation"],
            )
            analysis = analysis.model_copy(
                update={"alternative_model_recommendations": alternative}
            )
        else:
            for candidate in alternative.recommended_models:
                if not metric_evidence_ids.intersection(candidate.evidence_ids):
                    raise ValueError(
                        f"Alternative {candidate.model_name} must cite non-null performance metric evidence."
                    )
        return {
            "status": "complete",
            "provider": configuration["provider"],
            "model": model,
            "structured_output": analysis.model_dump(),
        }
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError, ValidationError) as exc:
        logger.warning("Structured LLM monitoring analysis failed: %s", exc)
        return {
            "status": "error",
            "provider": configuration["provider"],
            "model": model,
            "message": "LLM analysis could not be completed. Deterministic monitoring results are still available.",
            "structured_output": None,
        }
