import pandas as pd
import logging
from pathlib import Path
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.models import (MonitoredModel, Dataset, MonitoringRun, PerformanceMetric,
                               DriftResult, FeatureDrift, Alert, AgentAnalysis)
from app.services.model_artifacts import load_trusted_model
from app.services.validation import read_csv_upload, validate_columns
from app.monitoring.metrics import calculate_metrics, calculate_regression_metrics
from app.monitoring.drift import calculate_feature_drift
from app.services.health import evaluate_health, find_performance_degradations
from app.agents.agents import monitoring_agent, drift_agent, root_cause_agent, recommendation_agent
from app.agents.llm import analyze_monitoring_results, llm_status

logger = logging.getLogger(__name__)
router = APIRouter(tags=["monitoring"])
MAX_DATASET_BYTES = 50 * 1024 * 1024
PREVIEW_ROWS = 8


class MonitoringRequest(BaseModel):
    independent_features: list[str]
    target_feature: str


@router.post("/models/{model_id}/datasets")
async def upload_dataset(model_id: int, dataset_type: str = "reference",
                         file: UploadFile = File(...), db: Session = Depends(get_db)):
    model = db.get(MonitoredModel, model_id)
    if not model:
        raise HTTPException(404, "Model not found.")
    if dataset_type not in ("reference", "production"):
        raise HTTPException(422, "dataset_type must be reference or production.")
    content = await file.read(MAX_DATASET_BYTES + 1)
    if len(content) > MAX_DATASET_BYTES:
        raise HTTPException(413, "CSV file exceeds the 50 MB upload limit.")
    try:
        df = read_csv_upload(file.filename, content)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if not df.columns.is_unique:
        raise HTTPException(422, "CSV column names must be unique.")
    if not model.artifact_path:
        try:
            validate_columns(df, model.feature_columns + [model.target_column])
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
    dataset = Dataset(model_id=model_id, dataset_type=dataset_type, filename=Path(file.filename).name,
                      row_count=len(df), columns=list(df.columns),
                      content=df.to_csv(index=False))
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    response = {
        "id": dataset.id,
        "dataset_type": dataset_type,
        "filename": dataset.filename,
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
        "preview": df.head(PREVIEW_ROWS).where(pd.notna(df), None).to_dict(orient="records"),
    }
    if model.artifact_path:
        counterpart_type = "production" if dataset_type == "reference" else "reference"
        counterpart = db.query(Dataset).filter_by(model_id=model_id, dataset_type=counterpart_type).order_by(
            Dataset.id.desc()
        ).first()
        compatibility = None
        if counterpart:
            counterpart_columns = set(counterpart.columns)
            frame_columns = set(dataset.columns)
            reference_columns = frame_columns if dataset_type == "reference" else counterpart_columns
            production_columns = frame_columns if dataset_type == "production" else counterpart_columns
            compatibility = {
                "shared_columns": sorted(reference_columns & production_columns),
                "reference_only": sorted(reference_columns - production_columns),
                "production_only": sorted(production_columns - reference_columns),
            }
        response["compatibility"] = compatibility
    return response


@router.post("/models/{model_id}/monitor")
def run_monitoring(
    model_id: int,
    payload: MonitoringRequest | None = None,
    db: Session = Depends(get_db),
):
    model = db.get(MonitoredModel, model_id)
    if not model:
        raise HTTPException(404, "Model not found.")
    reference = db.query(Dataset).filter_by(model_id=model_id, dataset_type="reference").order_by(Dataset.id.desc()).first()
    production = db.query(Dataset).filter_by(model_id=model_id, dataset_type="production").order_by(Dataset.id.desc()).first()
    if not reference or not production:
        raise HTTPException(422, "Both reference and production datasets are required.")
    ref_df = pd.read_csv(pd.io.common.StringIO(reference.content))
    prod_df = pd.read_csv(pd.io.common.StringIO(production.content))

    if model.artifact_path:
        if not model.trusted_model:
            raise HTTPException(422, "The model file has not been confirmed as trusted.")
        if payload is None:
            raise HTTPException(422, "Select independent features and a target feature before monitoring.")
        features = payload.independent_features
        target = payload.target_feature
        if not features or len(set(features)) != len(features):
            raise HTTPException(422, "Select one or more unique independent features.")
        if target in features:
            raise HTTPException(422, "The target feature cannot also be an independent feature.")
        if target not in ref_df.columns:
            raise HTTPException(422, "The selected target column is missing from the original dataset.")
        missing = [column for column in features
                   if column not in ref_df.columns or column not in prod_df.columns]
        if missing:
            raise HTTPException(422, f"Selected features must exist in both datasets: {', '.join(missing)}")
        if len(prod_df) < 1:
            raise HTTPException(422, "The current dataset must contain at least one row.")
        if (model.model_feature_count is not None and not model.model_feature_names
                and len(features) != model.model_feature_count):
            raise HTTPException(422, f"This model requires {model.model_feature_count} input features.")
        try:
            trained_model = load_trusted_model(model.artifact_path)
            current_x = prod_df.loc[:, features]
            inference_x = current_x
            expected_names = model.model_feature_names
            if expected_names and list(current_x.columns) != list(expected_names):
                encoded_x = pd.get_dummies(current_x, columns=[
                    column for column in current_x.columns
                    if not pd.api.types.is_numeric_dtype(current_x[column])
                ], drop_first=True, dtype=float)
                if set(expected_names).issubset(encoded_x.columns):
                    inference_x = encoded_x.loc[:, expected_names]
                elif set(expected_names).issubset(current_x.columns):
                    inference_x = current_x.loc[:, expected_names]
                else:
                    missing_model_features = sorted(set(expected_names) - set(encoded_x.columns))
                    raise ValueError(
                        "Input columns do not match the trained model. "
                        f"Expected model inputs: {', '.join(expected_names)}. "
                        f"Still missing after one-hot encoding: {', '.join(missing_model_features)}."
                    )
            predictions = trained_model.predict(inference_x)
        except (ValueError, TypeError, KeyError) as exc:
            raise HTTPException(422, f"Model inference failed: {exc}") from exc
        except Exception as exc:
            logger.exception("Model inference failed for model id %s", model_id)
            raise HTTPException(422, "Model inference failed; check the selected feature types and model compatibility.") from exc
        if len(predictions) != len(prod_df):
            raise HTTPException(422, "The model returned a prediction count that does not match the current dataset.")
        task_type = "regression" if getattr(trained_model, "_estimator_type", None) == "regressor" else "classification"
        model.model_type = task_type
        metrics = (
            {"mae": None, "rmse": None, "r2": None}
            if task_type == "regression"
            else {"accuracy": None, "precision": None, "recall": None, "f1_score": None, "roc_auc": None}
        )
        if target in prod_df.columns:
            y_true = pd.to_numeric(prod_df[target], errors="coerce") if task_type == "regression" else prod_df[target]
            valid = y_true.notna()
            if valid.any():
                try:
                    if task_type == "regression":
                        metrics = calculate_regression_metrics(y_true[valid], predictions[valid])
                    else:
                        probabilities = None
                        classes = getattr(trained_model, "classes_", None)
                        try:
                            if callable(getattr(trained_model, "predict_proba", None)):
                                probabilities = trained_model.predict_proba(inference_x.loc[valid])
                            elif callable(getattr(trained_model, "decision_function", None)):
                                probabilities = trained_model.decision_function(inference_x.loc[valid])
                        except (ValueError, TypeError, AttributeError):
                            probabilities = None
                        metrics = calculate_metrics(y_true[valid], predictions[valid], probabilities, classes)
                except (ValueError, TypeError) as exc:
                    raise HTTPException(422, f"Could not calculate supervised metrics: {exc}") from exc
        model.target_column = target
        model.feature_columns = features
    else:
        target = model.target_column
        metrics = {"accuracy": None, "precision": None, "recall": None, "f1_score": None, "roc_auc": None}
        task_type = model.model_type.lower()
        supervised_available = target in ref_df and target in prod_df
        if supervised_available:
            y_true = pd.to_numeric(prod_df[target], errors="coerce")
            pred_col = "_prediction"
            if pred_col in prod_df:
                y_pred = pd.to_numeric(prod_df[pred_col], errors="coerce")
            elif "_predicted" in prod_df:
                y_pred = pd.to_numeric(prod_df["_predicted"], errors="coerce")
            else:
                y_pred = None
            if y_pred is not None:
                valid = y_true.notna() & y_pred.notna()
                metrics = calculate_metrics(y_true[valid], y_pred[valid],
                                           pd.to_numeric(prod_df["_probability"], errors="coerce")[valid].values
                                           if "_probability" in prod_df else None)
        features = model.feature_columns

    drift_config = model.threshold_config or {}
    psi_thresholds = {"warning": drift_config.get("psi_warning", 0.1),
                      "critical": drift_config.get("psi_critical", 0.25)}
    drift_features = calculate_feature_drift(ref_df, prod_df, features,
                                             drift_config.get("ks_alpha", 0.05), psi_thresholds)

    previous = db.query(MonitoringRun).filter_by(model_id=model_id).order_by(MonitoringRun.id.desc()).first()
    metric_payload = []
    for name, value in metrics.items():
        previous_value = None
        if previous:
            item = db.query(PerformanceMetric).filter_by(run_id=previous.id, name=name).first()
            if item:
                previous_value = item.value
        metric_payload.append({"name": name, "value": value, "previous_value": previous_value})

    health_config = {
        "metric_degradation": drift_config.get("metric_degradation", 0.05),
        "drift_warning": drift_config.get("drift_warning", 1),
        "drift_critical": drift_config.get("drift_critical", 3),
    }
    health_status, explanation = evaluate_health(metric_payload, drift_features, health_config)
    performance_degradations = find_performance_degradations(
        metric_payload, health_config["metric_degradation"]
    )
    run = MonitoringRun(model_id=model_id, reference_dataset_id=reference.id,
                        production_dataset_id=production.id,
                        health_status=health_status, health_explanation=explanation)
    db.add(run)
    db.flush()
    for item in metric_payload:
        db.add(PerformanceMetric(run_id=run.id, name=item["name"], value=item["value"],
                                 previous_value=item["previous_value"]))
    drift_result = DriftResult(run_id=run.id,
                               overall_drift=any(f["drift_detected"] for f in drift_features),
                               drifted_features=sum(1 for f in drift_features if f["drift_detected"]),
                               summary=f"{sum(1 for f in drift_features if f['drift_detected'])} features show drift.")
    db.add(drift_result)
    db.flush()
    for item in drift_features:
        db.add(FeatureDrift(drift_result_id=drift_result.id, **item))

    analyses = {
        "Monitoring Agent": monitoring_agent(metric_payload, health_status),
        "Drift Detection Agent": drift_agent(drift_features),
        "Root Cause Analysis Agent": root_cause_agent(metric_payload, drift_features),
        "Recommendation Agent": recommendation_agent(health_status, metric_payload, drift_features),
        "LLM Analysis": analyze_monitoring_results(
            metric_payload,
            drift_features,
            health_status,
            explanation,
            model_context={
                "name": model.name,
                "algorithm": model.model_class.rsplit(".", 1)[-1] if model.model_class else "Unknown",
                "problem_type": task_type,
                "features": features,
                "fit_diagnostics_available": False,
            },
            performance_degradations=performance_degradations,
        ),
    }
    for name, output in analyses.items():
        db.add(AgentAnalysis(run_id=run.id, agent_name=name, output=output))

    if health_status != "Healthy":
        db.add(Alert(model_id=model_id, run_id=run.id, severity="Critical" if health_status == "Critical" else "Warning",
                     title=f"{model.name} requires attention",
                     message=explanation))
    model.status = health_status
    db.commit()
    db.refresh(run)
    return {
        "id": run.id, "health_status": health_status, "health_explanation": explanation,
        "model_type": model.model_type,
        "status_factors": {
            "performance_metrics_available": any(value is not None for value in metrics.values()),
            "performance_metric_count": sum(value is not None for value in metrics.values()),
            "drift_features_checked": len(drift_features),
            "drifted_features": [feature["feature"] for feature in drift_features if feature["drift_detected"]],
            "critical_drift_features": [feature["feature"] for feature in drift_features if feature["severity"] == "Critical"],
            "warning_drift_features": [feature["feature"] for feature in drift_features if feature["severity"] == "Warning"],
        },
        "metrics": metric_payload,
        "drift": {"overall_drift": drift_result.overall_drift,
                  "drifted_features": drift_result.drifted_features,
                  "features": drift_features},
        "agents": analyses,
    }


@router.get("/models/{model_id}/runs")
def monitoring_runs(model_id: int, db: Session = Depends(get_db)):
    runs = db.query(MonitoringRun).filter_by(model_id=model_id).order_by(MonitoringRun.id.desc()).all()
    result = []
    for run in runs:
        model = db.get(MonitoredModel, run.model_id)
        metrics = [{"name": m.name, "value": m.value, "previous_value": m.previous_value}
                   for m in run.metrics]
        drift = None
        if run.drift:
            drift = {"overall_drift": run.drift.overall_drift,
                     "drifted_features": run.drift.drifted_features,
                     "features": [{
                         "feature": f.feature, "ks_statistic": f.ks_statistic,
                         "ks_p_value": f.ks_p_value, "psi": f.psi,
                         "drift_detected": f.drift_detected, "severity": f.severity,
                         "explanation": f.explanation} for f in run.drift.features]}
        agents = {a.agent_name: a.output for a in run.analyses}
        result.append({"id": run.id, "health_status": run.health_status,
                       "health_explanation": run.health_explanation,
                       "model_type": model.model_type if model else None,
                       "created_at": run.created_at.isoformat(), "metrics": metrics,
                       "drift": drift, "agents": agents,
                       "status_factors": {
                           "performance_metrics_available": any(item["value"] is not None for item in metrics),
                           "performance_metric_count": sum(item["value"] is not None for item in metrics),
                           "drift_features_checked": len(drift["features"]) if drift else 0,
                           "drifted_features": [
                               item["feature"] for item in (drift["features"] if drift else [])
                               if item["drift_detected"]
                           ],
                           "critical_drift_features": [
                               item["feature"] for item in (drift["features"] if drift else [])
                               if item["severity"] == "Critical"
                           ],
                           "warning_drift_features": [
                               item["feature"] for item in (drift["features"] if drift else [])
                               if item["severity"] == "Warning"
                           ],
                       }})
    return result


@router.get("/llm/status")
def get_llm_status():
    return llm_status()
