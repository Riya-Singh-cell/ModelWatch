# ModelWatch Architecture

The React frontend communicates with a modular FastAPI backend over REST.

```text
frontend (React/TypeScript/Recharts)
             |
             v
FastAPI routers
             |
  +----------+-----------+
  |                      |
SQLAlchemy/SQLite    Monitoring services
                         |
               Metrics + KS/PSI + Health
                         |
             Four deterministic agents
```

Database entities include monitored models, datasets, monitoring runs, performance metrics,
feature and aggregate drift results, alerts, and agent analyses.

Core decisions remain deterministic. Agent outputs summarize actual run evidence and do not
claim causal relationships.
