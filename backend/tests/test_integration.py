from app.monitoring.metrics import calculate_metrics
from app.monitoring.drift import calculate_feature_drift
from app.services.health import evaluate_health
from app.agents.agents import monitoring_agent, drift_agent, root_cause_agent, recommendation_agent
import pandas as pd


def test_agent_workflow_and_health_integration():
    metrics = calculate_metrics([0, 1, 1, 1], [0, 0, 1, 1])
    payload = [{"name": name, "value": value, "previous_value": 0.9 if name == "accuracy" else None}
               for name, value in metrics.items()]
    drift = calculate_feature_drift(pd.DataFrame({"age": range(10)}),
                                    pd.DataFrame({"age": range(50, 60)}), ["age"])
    status, explanation = evaluate_health(payload, drift)
    assert monitoring_agent(payload, status)["summary"]
    assert drift_agent(drift)["summary"]
    assert root_cause_agent(payload, drift)["confidence"]
    assert recommendation_agent(status, payload, drift)["recommendations"]
    assert explanation
