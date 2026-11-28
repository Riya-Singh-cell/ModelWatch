import { useEffect, useMemo, useState, type FormEvent } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid,
} from 'recharts'
import './App.css'

const api = axios.create({ baseURL: '/api' })
type DatasetType = 'reference' | 'production'
type Model = {
  id: number
  name: string
  model_type: string
  description: string
  target_column: string
  feature_columns: string[]
  status: string
  original_model_filename?: string
  model_class?: string
  model_feature_names?: string[] | null
  model_feature_count?: number | null
  trusted_model?: boolean
  health_explanation?: string | null
  latest_run_id?: number | null
  last_monitored_at?: string | null
}
type DatasetInfo = {
  id: number
  dataset_type: DatasetType
  filename: string
  row_count: number
  column_count: number
  columns: string[]
  preview: Record<string, unknown>[]
  compatibility?: {
    shared_columns: string[]
    reference_only: string[]
    production_only: string[]
  } | null
}
type Metric = { name: string; value: number | null; previous_value: number | null }
type DriftFeature = {
  feature: string
  ks_statistic: number | null
  ks_p_value: number | null
  psi: number | null
  drift_detected: boolean
  severity: string
  explanation: string
}
type Run = {
  id: number
  health_status: string
  health_explanation: string
  created_at: string
  metrics: Metric[]
  drift?: { overall_drift: boolean; drifted_features: number; features: DriftFeature[] }
  agents?: Record<string, unknown>
  model_type?: string
  status_factors?: {
    performance_metrics_available: boolean
    performance_metric_count: number
    drift_features_checked: number
    drifted_features: string[]
    critical_drift_features: string[]
    warning_drift_features: string[]
  }
}
type LLMAnalysis = {
  status: 'complete' | 'unconfigured' | 'error'
  provider: string
  model: string | null
  message?: string
  structured_output?: {
    summary: string
    findings: { severity: string; title: string; explanation: string; evidence_ids: string[] }[]
    possible_causes: { hypothesis: string; evidence_ids: string[]; confidence: string; caveat: string }[]
    recommendations: { priority: number; action: string; rationale: string; evidence_ids: string[] }[]
    alternative_model_recommendations: {
      replacement_required: boolean
      reason: string
      recommended_models: {
        model_name: string
        priority: number
        reason: string
        expected_advantage: string
        evaluation_required: boolean
        evidence_ids: string[]
      }[]
      next_steps: string[]
      evidence_ids: string[]
    }
    limitations: string[]
  } | null
}

function messageFromError(error: unknown) {
  if (axios.isAxiosError(error)) {
    const detail = error.response?.data?.detail
    return typeof detail === 'string' ? detail : error.message
  }
  return error instanceof Error ? error.message : 'Something went wrong.'
}

function recommendationsFor(run: Run | undefined) {
  const analysis = run?.agents?.['Recommendation Agent'] as
    | { recommendations?: { priority: number; action: string }[] }
    | undefined
  return analysis?.recommendations ?? []
}

function modelDisplayName(modelClass: string | undefined, fallback: string) {
  const className = modelClass?.split('.').pop() || fallback
  return className.replace(/([a-z0-9])([A-Z])/g, '$1 $2')
}

function LLMAnalysisCard({
  analysis,
  currentModelName,
  problemType,
}: {
  analysis: LLMAnalysis | undefined
  currentModelName: string
  problemType: string
}) {
  if (!analysis || analysis.status !== 'complete' || !analysis.structured_output) {
    const message = analysis?.message ?? 'No LLM analysis was saved for this run.'
    return (
      <section className="panel llm-panel">
        <div className="llm-heading"><div><span className="llm-kicker">AI-POWERED EXPLANATION</span><h2>Structured LLM analysis</h2></div>
          <span className={`status ${!analysis || analysis.status === 'error' ? 'warning' : 'not-monitored'}`}>
            {!analysis || analysis.status === 'error' ? 'Unavailable' : 'Not configured'}
          </span>
        </div>
        <p>{message}</p>
        <p className="switch-hint">Configure <code>LLM_API_KEY</code> and <code>LLM_MODEL</code> in the backend environment, then restart the backend. Metric, drift, health, and alert decisions continue to run deterministically.</p>
      </section>
    )
  }
  const output = analysis.structured_output
  return (
    <section className="panel llm-panel">
      <div className="llm-heading"><div><span className="llm-kicker">AI-POWERED EXPLANATION</span><h2>Structured LLM analysis</h2></div>
        <span className="status healthy">{analysis.model} · validated JSON</span>
      </div>
      <p className="llm-summary">{output.summary}</p>
      <div className="llm-columns">
        <div><h3>Findings</h3>
          {output.findings.length ? output.findings.map((finding, index) =>
            <article className="llm-item" key={`${finding.title}-${index}`}>
              <span className={`status ${finding.severity}`}>{finding.severity}</span>
              <strong>{finding.title}</strong><p>{finding.explanation}</p>
              {!!finding.evidence_ids.length && <small>Evidence: {finding.evidence_ids.join(', ')}</small>}
            </article>) : <p className="empty">No material findings.</p>}
        </div>
        <div><h3>Possible causes <small>(hypotheses, not proven causes)</small></h3>
          {output.possible_causes.length ? output.possible_causes.map((cause, index) =>
            <article className="llm-item" key={`${cause.hypothesis}-${index}`}>
              <strong>{cause.hypothesis}</strong><p>Confidence: {cause.confidence}. {cause.caveat}</p>
              {!!cause.evidence_ids.length && <small>Evidence: {cause.evidence_ids.join(', ')}</small>}
            </article>) : <p className="empty">No possible causes identified from available evidence.</p>}
        </div>
      </div>
      <div className="recommendation-list"><h3>Prioritized recommendations</h3>
        {output.recommendations.map(item => <article className="llm-item" key={`${item.priority}-${item.action}`}>
          <strong>Priority {item.priority}: {item.action}</strong><p>{item.rationale}</p>
          {!!item.evidence_ids.length && <small>Evidence: {item.evidence_ids.join(', ')}</small>}
        </article>)}
      </div>
      <section className="alternative-models">
        <div className="alternative-heading">
          <div>
            <h3>AI-Recommended Alternative Models</h3>
            <p>Current model: <strong>{currentModelName}</strong> · {problemType}</p>
          </div>
          <span className={`status ${output.alternative_model_recommendations.replacement_required ? 'warning' : 'healthy'}`}>
            {output.alternative_model_recommendations.replacement_required
              ? 'Evaluate alternatives'
              : 'No alternative recommended'}
          </span>
        </div>
        <p>{output.alternative_model_recommendations.reason}</p>
        {output.alternative_model_recommendations.recommended_models.length > 0 &&
          <div className="alternative-grid">
            {output.alternative_model_recommendations.recommended_models.map(candidate =>
              <article className="alternative-card" key={`${candidate.priority}-${candidate.model_name}`}>
                <div className="alternative-card-heading">
                  <span className="alternative-priority">Priority {candidate.priority}</span>
                  <span className="status warning">Evaluation required</span>
                </div>
                <h4>{candidate.model_name}</h4>
                <p><strong>Why consider it:</strong> {candidate.reason}</p>
                <p><strong>Potential advantage:</strong> {candidate.expected_advantage}</p>
                <small>Evidence: {candidate.evidence_ids.join(', ')}</small>
              </article>
            )}
          </div>}
        <div className="alternative-next-steps">
          <strong>Suggested evaluation strategy</strong>
          <ol>{output.alternative_model_recommendations.next_steps.map((step, index) =>
            <li key={`${index}-${step}`}>{step}</li>)}</ol>
        </div>
        <p className="switch-hint">Recommendations are candidates to validate, not production replacements. Compare them on the same labeled validation data before making a change.</p>
      </section>
      {!!output.limitations.length && <div className="llm-limitations"><strong>Limitations</strong>
        <ul>{output.limitations.map((limitation, index) => <li key={`${index}-${limitation}`}>{limitation}</li>)}</ul>
      </div>}
      <p className="switch-hint">The LLM receives model metadata, selected feature names, and aggregate metric and drift evidence—not dataset rows, the trained model file, or credentials. Monitoring decisions remain deterministic.</p>
    </section>
  )
}

function DatasetPreview({ title, data }: { title: string; data: DatasetInfo | null }) {
  if (!data) {
    return <div className="preview-empty">Upload a CSV to inspect its columns and sample rows.</div>
  }
  return (
    <div className="dataset-preview">
      <div className="dataset-summary">
        <strong>{data.filename}</strong>
        <span>{data.row_count.toLocaleString()} rows · {data.column_count} columns</span>
      </div>
      <div className="column-list">
        {data.columns.map(column => <span className="column-chip" key={`${title}-${column}`}>{column}</span>)}
      </div>
      <div className="table-scroll">
        <table className="preview-table">
          <thead><tr>{data.columns.map(column => <th key={column}>{column}</th>)}</tr></thead>
          <tbody>{data.preview.map((row, index) => <tr key={index}>
            {data.columns.map(column => <td key={column}>{String(row[column] ?? '—')}</td>)}
          </tr>)}</tbody>
        </table>
      </div>
    </div>
  )
}

function App() {
  const [page, setPage] = useState('Overview')
  const [dark, setDark] = useState(true)
  const [dashboard, setDashboard] = useState<Record<string, any>>({})
  const [models, setModels] = useState<Model[]>([])
  const [selected, setSelected] = useState<number | null>(null)
  const [runs, setRuns] = useState<Run[]>([])
  const [alerts, setAlerts] = useState<Record<string, any>[]>([])
  const [llmStatus, setLlmStatus] = useState<{ configured: boolean; provider: string; model: string | null; message?: string } | null>(null)
  const [datasets, setDatasets] = useState<Record<DatasetType, DatasetInfo | null>>({
    reference: null,
    production: null,
  })
  const [trustedSource, setTrustedSource] = useState(false)
  const [independentFeatures, setIndependentFeatures] = useState<string[]>([])
  const [targetFeature, setTargetFeature] = useState('')
  const [uploadProgress, setUploadProgress] = useState<Record<string, number | null>>({
    model: null, reference: null, production: null,
  })
  const [busy, setBusy] = useState(false)
  const [errorMessage, setErrorMessage] = useState('')
  const [successMessage, setSuccessMessage] = useState('')

  const selectedModel = models.find(model => model.id === selected)
  const commonColumns = useMemo(() => {
    if (!datasets.reference || !datasets.production) return []
    return datasets.reference.columns.filter(column => datasets.production?.columns.includes(column))
  }, [datasets])
  const availableFeatures = commonColumns.filter(column => column !== targetFeature)
  const latest = runs[0]
  const selectedRecommendations = recommendationsFor(latest)
  const driftFeatures = latest?.drift?.features ?? []
  const trend = runs.slice().reverse().map((run, index) => ({
    name: `Run ${index + 1}`,
    drift: run.drift?.drifted_features ?? 0,
  }))

  const refresh = async () => {
    const [dash, modelList, alertList, llm] = await Promise.all([
      api.get('/dashboard'), api.get('/models'), api.get('/alerts'), api.get('/llm/status'),
    ])
    setDashboard(dash.data)
    setModels(modelList.data)
    setAlerts(alertList.data)
    setLlmStatus(llm.data)
  }

  useEffect(() => {
    refresh().catch(error => setErrorMessage(`Could not load ModelWatch data: ${messageFromError(error)}`))
  }, [])

  useEffect(() => {
    if (!selected) {
      setRuns([])
      setDatasets({ reference: null, production: null })
      setIndependentFeatures([])
      setTargetFeature('')
      return
    }
    let active = true
    Promise.all([
      api.get(`/models/${selected}/runs`),
      api.get(`/models/${selected}/datasets`).catch(() => ({ data: [] })),
    ]).then(([runResponse, datasetResponse]) => {
      if (!active) return
      setRuns(runResponse.data)
      const data: Record<DatasetType, DatasetInfo | null> = { reference: null, production: null }
      for (const item of datasetResponse.data as DatasetInfo[]) data[item.dataset_type] = item
      setDatasets(data)
      setIndependentFeatures([])
      setTargetFeature('')
    }).catch(error => {
      if (active) setErrorMessage(`Could not load the selected model: ${messageFromError(error)}`)
    })
    return () => { active = false }
  }, [selected])

  useEffect(() => {
    if (!selectedModel || !datasets.reference || !datasets.production) return
    const shared = datasets.reference.columns.filter(column => datasets.production?.columns.includes(column))
    const expected = selectedModel.model_feature_names ?? []
    let suggestedFeatures: string[] = []
    if (expected.length && expected.every(name => shared.includes(name))) {
      suggestedFeatures = expected
    } else if (expected.length) {
      suggestedFeatures = expected.map(name => {
        if (shared.includes(name)) return name
        return shared.find(column => name.startsWith(`${column}_`)) ?? ''
      }).filter(Boolean)
    }
    if (!suggestedFeatures.length && selectedModel.model_feature_count &&
        shared.length === selectedModel.model_feature_count + 1) {
      suggestedFeatures = shared
    }
    const distinctFeatures = Array.from(new Set(suggestedFeatures))
    const targetCandidates = shared.filter(column => !distinctFeatures.includes(column))
    if (distinctFeatures.length && distinctFeatures.length <= shared.length) {
      setIndependentFeatures(distinctFeatures)
    }
    if (!targetFeature && targetCandidates.length === 1) {
      setTargetFeature(targetCandidates[0])
    }
  // Only re-suggest values when the workspace or one of its uploaded datasets changes.
  }, [selected, datasets.reference?.id, datasets.production?.id])

  const uploadModel = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    const form = event.currentTarget
    setErrorMessage('')
    setSuccessMessage('')
    const file = (form.elements.namedItem('model-file') as HTMLInputElement).files?.[0]
    if (!file) return setErrorMessage('Choose a trained .pkl model file.')
    if (!file.name.toLowerCase().endsWith('.pkl')) return setErrorMessage('Only .pkl model files are accepted.')
    if (!trustedSource) return setErrorMessage('Confirm that the model file comes from a trusted source.')
    const formData = new FormData()
    formData.append('file', file)
    formData.append('trusted_source', 'true')
    setBusy(true)
    try {
      const response = await api.post('/model-workspaces', formData, {
        onUploadProgress: progress => setUploadProgress(value => ({
          ...value, model: progress.total ? Math.round(progress.loaded * 100 / progress.total) : null,
        })),
      })
      await refresh()
      setSelected(response.data.id)
      setDatasets({ reference: null, production: null })
      setSuccessMessage(`Model uploaded: ${response.data.model_class}`)
      form.reset()
      setTrustedSource(false)
    } catch (error) {
      setErrorMessage(`Model upload failed: ${messageFromError(error)}`)
    } finally {
      setBusy(false)
      setUploadProgress(value => ({ ...value, model: null }))
    }
  }

  const uploadDataset = async (datasetType: DatasetType, file?: File) => {
    setErrorMessage('')
    setSuccessMessage('')
    if (!selected) return setErrorMessage('Upload or select a trained model first.')
    if (!file) return
    if (!file.name.toLowerCase().endsWith('.csv')) return setErrorMessage('Only CSV dataset files are accepted.')
    const formData = new FormData()
    formData.append('file', file)
    try {
      const response = await api.post(`/models/${selected}/datasets?dataset_type=${datasetType}`, formData, {
        onUploadProgress: progress => setUploadProgress(value => ({
          ...value, [datasetType]: progress.total ? Math.round(progress.loaded * 100 / progress.total) : null,
        })),
      })
      setDatasets(value => ({ ...value, [datasetType]: response.data }))
      setIndependentFeatures([])
      setTargetFeature('')
      setSuccessMessage(`${datasetType === 'reference' ? 'Original' : 'Current'} dataset uploaded and validated.`)
    } catch (error) {
      setErrorMessage(`Dataset upload failed: ${messageFromError(error)}`)
    } finally {
      setUploadProgress(value => ({ ...value, [datasetType]: null }))
    }
  }

  const monitor = async () => {
    if (!selected) return
    setBusy(true)
    setErrorMessage('')
    setSuccessMessage('')
    try {
      await api.post(`/models/${selected}/monitor`, {
        independent_features: independentFeatures,
        target_feature: targetFeature,
      })
      const [runResponse, alertResponse] = await Promise.all([
        api.get(`/models/${selected}/runs`), api.get('/alerts'),
      ])
      setRuns(runResponse.data)
      setAlerts(alertResponse.data)
      await refresh()
      setPage('Monitoring')
      setSuccessMessage('Monitoring completed successfully.')
    } catch (error) {
      setErrorMessage(`Monitoring failed: ${messageFromError(error)}`)
    } finally {
      setBusy(false)
    }
  }

  const cards = [
    ['Total models', dashboard.total_models ?? 0],
    ['Healthy', dashboard.healthy_models ?? 0],
    ['Warning', dashboard.warning_models ?? 0],
    ['Critical', dashboard.critical_models ?? 0],
    ['Not monitored', dashboard.unmonitored_models ?? 0],
    ['Insufficient data', dashboard.insufficient_data_models ?? 0],
  ]

  return (
    <div className={dark ? 'app dark' : 'app'}>
      <aside>
        <div className="brand"><span>MW</span> ModelWatch</div>
        {['Overview', 'Models', 'Monitoring', 'Data Drift', 'Alerts'].map(item =>
          <button key={item} className={page === item ? 'nav active' : 'nav'} onClick={() => setPage(item)}>
            {item}
          </button>)}
      </aside>
      <main>
        <header>
          <div><h1>{page}</h1><p>ML monitoring and drift detection</p></div>
          <button className="theme" onClick={() => setDark(value => !value)}>{dark ? 'Light' : 'Dark'} theme</button>
        </header>
        {errorMessage && <div className="notice error" role="alert">{errorMessage}<button onClick={() => setErrorMessage('')}>×</button></div>}
        {successMessage && <div className="notice success" role="status">{successMessage}<button onClick={() => setSuccessMessage('')}>×</button></div>}

        {page === 'Overview' && <>
          <div className={`overall-banner ${String(dashboard.overall_status ?? 'not-monitored').toLowerCase().replaceAll(' ', '-')}`}>
            <span>Overall model health</span>
            <strong>{dashboard.overall_status ?? 'Loading'}</strong>
            {dashboard.unmonitored_models > 0 &&
              <small>{dashboard.unmonitored_models} model(s) need a monitoring run before health can be assessed.</small>}
          </div>
          <section className="cards">{cards.map(([label, value]) =>
            <div className="card" key={label as string}><span>{label}</span><strong>{value}</strong></div>)}</section>
          <section className="grid-two">
            <div className="panel"><h2>Drift trend</h2>
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={trend}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="name" /><YAxis /><Tooltip /><Line dataKey="drift" stroke="#4f46e5" strokeWidth={3} /></LineChart>
              </ResponsiveContainer>
            </div>
            <div className="panel"><h2>Recent activity</h2>
              {(dashboard.recent_activity ?? []).map((item: any) => <div className="activity" key={item.run_id}>
                <span className={`status ${item.status.toLowerCase()}`}>{item.status}</span>
                <div><strong>{item.model_name}</strong><small>{new Date(item.created_at).toLocaleString()}</small></div>
              </div>)}
              {!(dashboard.recent_activity?.length) && <p className="empty">No monitoring runs yet.</p>}
            </div>
          </section>
        </>}

        {page === 'Models' && <>
          <section className="panel workspace-panel">
            <div className="section-heading">
              <div><h2>Monitor a trained model</h2><p>Upload a trusted model and its original and production datasets.</p></div>
              <div className="model-picker"><label htmlFor="model-picker">Existing model workspace</label>
                <select id="model-picker" value={selected ?? ''} onChange={event => setSelected(event.target.value ? Number(event.target.value) : null)}>
                  <option value="">Select a model</option>
                  {models.filter(model => model.original_model_filename).map(model =>
                    <option key={model.id} value={model.id}>
                      {model.name} · {model.model_class ?? model.model_type} (#{model.id})
                    </option>)}
                </select>
              </div>
            </div>

            <form onSubmit={uploadModel} className="model-upload">
              <div className="upload-title"><span className="step">01</span><div><h3>Upload Trained ML Model</h3><p>Only trusted scikit-learn/joblib-compatible .pkl models are supported.</p></div></div>
              <input name="model-file" type="file" accept=".pkl" aria-label="Upload Trained ML Model" disabled={busy} />
              <label className="trust-check"><input type="checkbox" checked={trustedSource} onChange={event => setTrustedSource(event.target.checked)} />
                I trust the source of this model file and understand that loading pickle files can execute code.</label>
              <button type="submit" disabled={!trustedSource || busy}>{busy ? 'Uploading…' : 'Upload model'}</button>
              {uploadProgress.model !== null && <progress max="100" value={uploadProgress.model ?? 0} />}
              {selectedModel?.model_class && <p className="validation-ok">Loaded model: {selectedModel.model_class}
                {selectedModel.model_feature_count ? ` · ${selectedModel.model_feature_count} expected features` : ''}</p>}
            </form>

            <div className="dataset-grid">
              {([
                ['reference', '02', 'Upload Original Dataset (Reference Data)'],
                ['production', '03', 'Upload Current Dataset (Production Data)'],
              ] as const).map(([type, step, title]) => (
                <section className="dataset-card" key={type}>
                  <div className="upload-title"><span className="step">{step}</span><div><h3>{title}</h3>
                    <p>{type === 'reference' ? 'Training or baseline evaluation data.' : 'Recent real-world production observations.'}</p>
                  </div></div>
                  <label className="file-control">{datasets[type] ? 'Replace CSV' : 'Choose CSV'}
                    <input type="file" accept=".csv,text/csv" disabled={!selected || busy}
                      aria-label={title} onChange={event => {
                        const file = event.currentTarget.files?.[0]
                        void uploadDataset(type, file)
                        event.currentTarget.value = ''
                      }} />
                  </label>
                  {uploadProgress[type] !== null && <progress max="100" value={uploadProgress[type] ?? 0} />}
                  <DatasetPreview title={title} data={datasets[type]} />
                </section>
              ))}
            </div>

            {datasets.reference && datasets.production && <section className="feature-section">
              <div className="upload-title"><span className="step">04</span><div><h3>Select model features</h3>
                <p>Input features must be available in both datasets. The target may be absent from production data.</p>
              </div></div>
              <div className="compatibility">
                <span className="validation-ok">Datasets loaded · {commonColumns.length} shared columns</span>
                {datasets.production.compatibility && (datasets.production.compatibility.reference_only.length > 0) &&
                  <span className="validation-warning">Current data lacks: {datasets.production.compatibility.reference_only.join(', ')}</span>}
              </div>
              <div className="feature-fields">
                <label>Independent Features (X)
                  <select multiple value={independentFeatures} onChange={event => {
                    setIndependentFeatures(Array.from(event.currentTarget.selectedOptions, option => option.value))
                  }}>
                    {availableFeatures.map(column => <option key={column} value={column}>{column}</option>)}
                  </select>
                  <small>Use Ctrl/⌘ to select multiple columns.</small>
                </label>
                <label>Target Feature (y)
                  <select value={targetFeature} onChange={event => {
                    const target = event.target.value
                    setTargetFeature(target)
                    setIndependentFeatures(features => features.filter(feature => feature !== target))
                  }}>
                    <option value="">Select target column</option>
                    {datasets.reference.columns.map(column => <option key={column} value={column}>{column}</option>)}
                  </select>
                  {targetFeature && !datasets.production.columns.includes(targetFeature) &&
                    <small className="validation-warning">Not present in production; supervised metrics will be unavailable.</small>}
                </label>
              </div>
              {selectedModel?.model_feature_names?.length &&
                <p className="model-feature-hint">Model feature names: {selectedModel.model_feature_names.join(', ')}</p>}
              <button className="start-monitoring" disabled={busy || !selected || !datasets.reference || !datasets.production ||
                independentFeatures.length === 0 || !targetFeature || independentFeatures.includes(targetFeature) ||
                independentFeatures.some(feature => !commonColumns.includes(feature))}
                onClick={monitor}>{busy ? 'Monitoring…' : 'Start Model Monitoring'}</button>
            </section>}
          </section>

          <section className="panel"><h2>Model workspaces</h2>
            <div className="table-scroll"><table><thead><tr><th>Model</th><th>Class</th><th>Features</th><th>Status</th></tr></thead>
              <tbody>{models.filter(model => model.original_model_filename).map(model => <tr key={model.id}
                className={selected === model.id ? 'selected-row' : ''} onClick={() => setSelected(model.id)}>
                <td><strong>{model.original_model_filename}</strong><small className="table-subtitle">{model.name} · workspace #{model.id}</small></td>
                <td>{model.model_class ?? model.model_type}</td>
                <td>{model.model_feature_count ?? model.model_feature_names?.length ?? 'Not reported'}</td>
                <td><span className={`status ${model.status.toLowerCase().replaceAll(' ', '-')}`}>{model.status}</span>
                  {model.health_explanation && <small className="table-subtitle">{model.health_explanation}</small>}
                  {!model.latest_run_id && <small className="table-subtitle">Upload datasets and run monitoring to get a health assessment.</small>}
                </td>
              </tr>)}</tbody>
            </table></div>
            {!models.some(model => model.original_model_filename) && <p className="empty">No uploaded model workspaces yet.</p>}
            {selectedModel && <div className="selected-model-detail">
              <h3>Selected model details</h3>
              <p><strong>{selectedModel.original_model_filename}</strong> · {selectedModel.model_type} model · workspace #{selectedModel.id}</p>
              <p>Estimator: {selectedModel.model_class ?? 'Unknown'} · Expected input features: {selectedModel.model_feature_count ?? 'Not reported'}</p>
              <p>Status: <span className={`status ${selectedModel.status.toLowerCase().replaceAll(' ', '-')}`}>{selectedModel.status}</span></p>
              <p>{selectedModel.health_explanation ?? 'This model has not been monitored yet. Upload both datasets and run monitoring to determine its health.'}</p>
              {selectedModel.latest_run_id && <button className="secondary-button" onClick={() => setPage('Monitoring')}>View monitoring details</button>}
            </div>}
          </section>
        </>}

        {page === 'Monitoring' && <>
          <section className="panel control">
            <div><label>Selected model</label><select value={selected ?? ''} onChange={event => setSelected(event.target.value ? Number(event.target.value) : null)}>
              <option value="">Choose a model</option>{models.map(model =>
                <option value={model.id} key={model.id}>{model.name} — {model.status} (#{model.id})</option>)}
            </select></div>
            {llmStatus && <span className={`status ${llmStatus.configured ? 'healthy' : 'not-monitored'}`}>
              {llmStatus.configured ? `LLM connected · ${llmStatus.model}` : 'LLM not configured'}
            </span>}
            <button onClick={monitor} disabled={!selected || busy}>{busy ? 'Monitoring…' : 'Run monitoring'}</button>
          </section>
          {latest ? <>
            <section className="panel health-panel">
              <div className="health-heading">
                <div><span className={`status ${latest.health_status.toLowerCase().replaceAll(' ', '-')}`}>{latest.health_status}</span>
                  <h2>Model health assessment</h2>
                </div>
                <label className="model-picker">Review another model
                  <select value={selected ?? ''} onChange={event => setSelected(event.target.value ? Number(event.target.value) : null)}>
                    {models.filter(model => model.original_model_filename).map(model =>
                      <option key={model.id} value={model.id}>{model.name} — {model.status}</option>)}
                  </select>
                </label>
              </div>
              <p className="health-explanation">{latest.health_explanation}</p>
              <div className="health-facts">
                <div><strong>{latest.status_factors?.performance_metric_count ?? 0}</strong><span>performance metrics available</span></div>
                <div><strong>{latest.status_factors?.drift_features_checked ?? latest.drift?.features.length ?? 0}</strong><span>features checked for drift</span></div>
                <div><strong>{latest.status_factors?.drifted_features.length ?? latest.drift?.drifted_features ?? 0}</strong><span>features showing drift</span></div>
              </div>
              {!latest.status_factors?.performance_metrics_available &&
                <p className="health-caveat">Performance can’t be confirmed without valid production ground-truth labels. The status currently reflects feature drift only.</p>}
              {!!selectedRecommendations.length && <div className="recommendation-list">
                <h3>What to do next</h3>
                {selectedRecommendations.map(item => <p key={item.priority}><span>Priority {item.priority}</span>{item.action}</p>)}
              </div>}
              <p className="switch-hint">Switching models changes the selected workspace. A different model needs its own compatible feature mapping and monitoring results before health can be compared.</p>
            </section>
            <section className="cards">
              {latest.metrics.map(metric => <div className="card" key={metric.name}>
                <span>{metric.name.replaceAll('_', ' ').toUpperCase()}</span><strong>{metric.value == null ? 'Unavailable' : metric.value.toFixed(3)}</strong>
                {metric.previous_value != null && <small>Previous: {metric.previous_value.toFixed(3)}</small>}
                {(metric.name === 'rmse' || metric.name === 'mae') && <small>Lower is better</small>}
                {metric.name === 'r2' && <small>Higher is better; 1.0 is a perfect fit</small>}
              </div>)}
            </section>
            <section className="panel"><h2>Metric trend</h2>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={latest.metrics.filter(metric => metric.value !== null || metric.previous_value !== null)
                  .map(metric => ({ name: metric.name, current: metric.value ?? 0, previous: metric.previous_value ?? 0 }))}>
                  <CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="name" /><YAxis /><Tooltip /><Bar dataKey="previous" fill="#94a3b8" /><Bar dataKey="current" fill="#4f46e5" />
                </BarChart>
              </ResponsiveContainer>
            </section>
            <LLMAnalysisCard
              analysis={latest.agents?.['LLM Analysis'] as LLMAnalysis | undefined}
              currentModelName={modelDisplayName(selectedModel?.model_class, selectedModel?.name ?? 'Unknown model')}
              problemType={latest.model_type ?? selectedModel?.model_type ?? 'Unknown problem type'}
            />
            <section className="panel"><h2>AI monitoring report</h2>
              {latest.agents && Object.entries(latest.agents).map(([name, output]) =>
                <div className="agent" key={name}><strong>{name}</strong><pre>{JSON.stringify(output, null, 2)}</pre></div>)}
            </section>
          </> : <p className="empty panel">Select a model workspace with monitoring data to see its health, metrics, drift, and recommended next steps. If it has never been monitored, its status is Not monitored rather than Healthy.</p>}
        </>}

        {page === 'Data Drift' && <>
          <section className="panel control"><div><label>Selected model</label><select value={selected ?? ''} onChange={event => setSelected(event.target.value ? Number(event.target.value) : null)}>
            <option value="">Choose a model</option>{models.map(model =>
              <option value={model.id} key={model.id}>{model.name} — {model.status} (#{model.id})</option>)}</select></div></section>
          {driftFeatures.length ? <section className="panel"><h2>Feature drift analysis</h2>
            <div className="table-scroll"><table><thead><tr><th>Feature</th><th>KS statistic</th><th>p-value</th><th>PSI</th><th>Severity</th></tr></thead>
              <tbody>{driftFeatures.map(feature => <tr key={feature.feature}><td><strong>{feature.feature}</strong></td>
                <td>{feature.ks_statistic?.toFixed(4) ?? '—'}</td><td>{feature.ks_p_value?.toFixed(4) ?? '—'}</td>
                <td>{feature.psi?.toFixed(4) ?? '—'}</td><td><span className={`status ${feature.severity.toLowerCase()}`}>{feature.severity}</span></td></tr>)}</tbody>
            </table></div>
            {driftFeatures.map(feature => <p className="explanation" key={feature.feature}>{feature.explanation}</p>)}
          </section> : <p className="empty panel">Run monitoring to calculate drift results.</p>}
        </>}

        {page === 'Alerts' && <>
          <section className="panel"><h2>Alert history</h2>
            {alerts.map(alert => <div className="alert" key={alert.id}>
              <span className={`status ${alert.severity.toLowerCase()}`}>{alert.severity}</span>
              <div><strong>{alert.title}</strong><p>{alert.message}</p><small>{new Date(alert.created_at).toLocaleString()}</small></div>
              {!alert.acknowledged && <button onClick={() => api.post(`/alerts/${alert.id}/acknowledge`).then(refresh)
                .catch(error => setErrorMessage(messageFromError(error)))}>Acknowledge</button>}
            </div>)}
            {!alerts.length && <p className="empty">No alerts recorded.</p>}
          </section>
        </>}
      </main>
    </div>
  )
}

export default App
