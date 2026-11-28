from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.models import MonitoredModel
from app.models.models import MonitoringRun
from app.schemas.schemas import ModelCreate, ModelRead
from app.services.model_artifacts import remove_model_artifact

router = APIRouter(prefix="/models", tags=["models"])


@router.post("", response_model=ModelRead, status_code=201)
def create_model(payload: ModelCreate, db: Session = Depends(get_db)):
    if payload.target_column in payload.feature_columns:
        raise HTTPException(422, "Target column cannot also be a feature column.")
    model = MonitoredModel(**payload.model_dump())
    db.add(model)
    db.commit()
    db.refresh(model)
    return model


@router.get("", response_model=list[ModelRead])
def list_models(db: Session = Depends(get_db)):
    models = db.query(MonitoredModel).order_by(MonitoredModel.id.desc()).all()
    return [_model_summary(model, db) for model in models]


@router.get("/{model_id}", response_model=ModelRead)
def get_model(model_id: int, db: Session = Depends(get_db)):
    model = db.get(MonitoredModel, model_id)
    if not model:
        raise HTTPException(404, "Model not found.")
    return _model_summary(model, db)


def _model_summary(model: MonitoredModel, db: Session):
    latest = db.query(MonitoringRun).filter_by(model_id=model.id).order_by(MonitoringRun.id.desc()).first()
    return {
        "id": model.id,
        "name": model.name,
        "model_type": model.model_type,
        "description": model.description or "",
        "target_column": model.target_column,
        "feature_columns": model.feature_columns or [],
        "status": latest.health_status if latest else "Not monitored",
        "threshold_config": model.threshold_config or {},
        "original_model_filename": model.original_model_filename,
        "model_class": model.model_class,
        "model_feature_names": model.model_feature_names,
        "model_feature_count": model.model_feature_count,
        "trusted_model": model.trusted_model,
        "health_explanation": latest.health_explanation if latest else None,
        "latest_run_id": latest.id if latest else None,
        "last_monitored_at": latest.created_at.isoformat() if latest else None,
    }


@router.put("/{model_id}", response_model=ModelRead)
def update_model(model_id: int, payload: ModelCreate, db: Session = Depends(get_db)):
    model = db.get(MonitoredModel, model_id)
    if not model:
        raise HTTPException(404, "Model not found.")
    for key, value in payload.model_dump().items():
        setattr(model, key, value)
    db.commit()
    db.refresh(model)
    return model


@router.delete("/{model_id}", status_code=204)
def delete_model(model_id: int, db: Session = Depends(get_db)):
    model = db.get(MonitoredModel, model_id)
    if not model:
        raise HTTPException(404, "Model not found.")
    artifact_path = model.artifact_path
    db.delete(model)
    db.commit()
    remove_model_artifact(artifact_path)
