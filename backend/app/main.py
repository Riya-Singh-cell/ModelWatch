from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from app.database import get_db, initialize_database
from app.models import models as model_tables
from app.routers.models import router as models_router
from app.routers.monitoring import router as monitoring_router
from app.routers.alerts import router as alerts_router
from app.models.models import MonitoredModel, MonitoringRun, Alert

initialize_database()

app = FastAPI(title="ModelWatch API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])
app.include_router(models_router, prefix="/api")
app.include_router(monitoring_router, prefix="/api")
app.include_router(alerts_router, prefix="/api")
from app.routers.workspaces import router as workspaces_router
app.include_router(workspaces_router, prefix="/api")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/dashboard")
def dashboard(db=Depends(get_db)):
    models = db.query(MonitoredModel).all()
    runs = db.query(MonitoringRun).order_by(MonitoringRun.id.desc()).limit(20).all()
    alerts = db.query(Alert).filter_by(acknowledged=False).count()
    recent = []
    for run in runs[:8]:
        model = db.get(MonitoredModel, run.model_id)
        recent.append({"run_id": run.id, "model_name": model.name if model else "Unknown",
                       "status": run.health_status, "created_at": run.created_at.isoformat()})
    return {
        "total_models": len(models),
        "healthy_models": sum(1 for m in models if m.status == "Healthy"),
        "warning_models": sum(1 for m in models if m.status == "Warning"),
        "critical_models": sum(1 for m in models if m.status == "Critical"),
        "unmonitored_models": sum(1 for m in models if m.status == "Not monitored"),
        "insufficient_data_models": sum(1 for m in models if m.status == "Insufficient Data"),
        "models_with_drift": sum(1 for m in models if m.status in ("Warning", "Critical")),
        "requires_attention": sum(1 for m in models if m.status == "Critical"),
        "overall_status": "Critical" if any(m.status == "Critical" for m in models)
        else "Warning" if any(m.status == "Warning" for m in models)
        else "Insufficient Data" if any(m.status == "Insufficient Data" for m in models)
        else "Healthy" if any(m.status == "Healthy" for m in models)
        else "Not monitored",
        "open_alerts": alerts,
        "recent_activity": recent,
        "trend": [{"run_id": r.id, "status": r.health_status,
                   "drifted": r.drift.drifted_features if r.drift else 0} for r in reversed(runs)]
    }
