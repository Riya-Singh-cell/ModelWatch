from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.models import Alert

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.get("")
def list_alerts(model_id: int | None = None, db: Session = Depends(get_db)):
    query = db.query(Alert)
    if model_id is not None:
        query = query.filter_by(model_id=model_id)
    return [{"id": a.id, "model_id": a.model_id, "run_id": a.run_id,
             "severity": a.severity, "title": a.title, "message": a.message,
             "acknowledged": a.acknowledged, "created_at": a.created_at.isoformat()}
            for a in query.order_by(Alert.id.desc()).all()]


@router.post("/{alert_id}/acknowledge")
def acknowledge(alert_id: int, db: Session = Depends(get_db)):
    alert = db.get(Alert, alert_id)
    if not alert:
        raise HTTPException(404, "Alert not found.")
    alert.acknowledged = True
    db.commit()
    return {"id": alert.id, "acknowledged": True}
