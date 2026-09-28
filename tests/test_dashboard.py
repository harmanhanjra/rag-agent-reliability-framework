from streamlit.testing.v1 import AppTest


def test_dashboard_runs_benchmark_and_renders_metrics():
    at = AppTest.from_file("app.py")
    at.run(timeout=30)
    assert not at.exception
    assert at.title[0].value == "RAG Evaluation & Agent Reliability Framework"
    assert len(at.button) == 1

    at.button[0].click().run(timeout=30)
    assert not at.exception
    labels = [metric.label for metric in at.metric]
    values = [metric.value for metric in at.metric]
    assert "Task completion" in labels
    assert "RAG accuracy" in labels
    assert "100%" in values
