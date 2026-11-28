# ModelWatch

ModelWatch is a full-stack ML monitoring and drift detection platform. It provides model registration,
CSV dataset uploads, supervised metric calculation, KS and PSI drift analysis, health evaluation,
historical runs, alerts, and evidence-based AI agent summaries.

## Requirements

- Python 3.10+
- Node.js 20+

## Backend

```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API documentation: `http://localhost:8000/docs`. Keep this backend process running while using
the frontend; Vite forwards `/api` requests to it.

### Optional structured LLM analysis

ModelWatch can request a JSON-schema-constrained explanation from an OpenAI-compatible chat
completions endpoint. Set the values in `backend/.env` before starting the backend:

```dotenv
LLM_API_KEY=your-provider-api-key
LLM_MODEL=openai/gpt-oss-20b
LLM_BASE_URL=https://api.groq.com/openai/v1
```

The backend loads this file automatically. Restart the backend after editing it. The file is
git-ignored; do not commit it or share the API key. For Docker Compose, the backend service also
loads `backend/.env` automatically.

The key is never sent to the browser or stored in SQLite. LLM requests include model metadata,
selected feature names, aggregate metric values, health status, and drift statistics; raw dataset
rows and trained model files are not sent. The LLM cannot change deterministic health, alert, or
drift decisions. Its structured output is validated and persisted with the monitoring run. If it
is unconfigured or unavailable, the UI reports that state and deterministic monitoring results
remain available.

## Frontend

```powershell
cd frontend
npm install
npm run dev
```

Dashboard: `http://localhost:5173`

## Tests

```powershell
cd backend
.venv\Scripts\Activate.ps1
pytest -q
```

## Docker

```powershell
docker compose up --build
```

## Monitoring flow

1. In **Models**, upload a trusted `.pkl` estimator and confirm that you trust its source.
   Pickle/joblib files can execute code when loaded; do not upload files from unknown sources.
2. Upload the original and current CSV datasets. Their previews, dimensions, and shared columns
   are displayed in the workspace.
3. Select independent features and a target. If model feature names can be mapped to CSV columns,
   ModelWatch suggests them automatically. Categorical columns are one-hot encoded when needed to
   match the model's declared feature names.
4. Choose **Start Model Monitoring**. The app runs actual estimator predictions, feature drift tests,
   persists a monitoring run, updates health, and generates agent explanations and recommendations.
5. Select another uploaded model workspace to review its own data, last health result, and metrics.
   Each model must have its own compatible datasets and feature selection before comparison.

Classification estimators report accuracy, precision, recall, F1, and ROC-AUC where supported.
Regression estimators report MAE, RMSE, and R² instead; classification metrics do not apply to a
regression model. If production labels are missing, supervised metrics are marked unavailable and
the health explanation explicitly says the result is based on drift only.
New workspaces show **Not monitored** until a pipeline run completes. **Insufficient Data** means
there were no usable performance or drift signals. Warning/Critical statuses list the detected
signals and next actions in the Monitoring page.

The structured LLM report may suggest up to three task-compatible algorithms when labeled
performance evidence supports evaluating them. Drift alone never triggers an alternative-model
recommendation; without production labels, the report asks for labeled evaluation instead.
Suggested algorithms are evaluation candidates only and are never trained or promoted
automatically. Training-versus-validation diagnostics are not currently collected, so the LLM
does not infer overfitting or underfitting.

## API

- `POST /api/models`
- `GET /api/models`
- `PUT /api/models/{id}`
- `DELETE /api/models/{id}`
- `POST /api/models/{id}/datasets?dataset_type=reference|production`
- `POST /api/models/{id}/monitor`
- `GET /api/models/{id}/runs`
- `GET /api/alerts`
- `POST /api/alerts/{id}/acknowledge`
- `GET /api/dashboard`
