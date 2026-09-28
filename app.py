import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from rag_eval import run_benchmark

st.set_page_config(page_title="RAG & Agent Reliability", layout="wide")
st.title("RAG Evaluation & Agent Reliability Framework")
st.caption("LlamaIndex retrieval contracts + LangChain structured tools · deterministic offline benchmark")

if st.button("Run benchmark", type="primary"):
    report = run_benchmark()
    Path("reports").mkdir(exist_ok=True)
    Path("reports/latest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    st.session_state["report"] = report

report = st.session_state.get("report")
if report:
    base = report["baseline"]["metrics"]
    hard = report["hardened"]["metrics"]
    cols = st.columns(5)
    cols[0].metric("Task completion", f"{hard['task_completion_rate']:.0%}", f"{hard['task_completion_rate']-base['task_completion_rate']:+.0%}")
    cols[1].metric("RAG accuracy", f"{hard['rag_answer_accuracy']:.0%}", f"{hard['rag_answer_accuracy']-base['rag_answer_accuracy']:+.0%}")
    cols[2].metric("Groundedness", f"{hard['groundedness_rate']:.0%}", f"{hard['groundedness_rate']-base['groundedness_rate']:+.0%}")
    cols[3].metric("Tool reliability", f"{hard['tool_reliability']:.0%}", f"{hard['tool_reliability']-base['tool_reliability']:+.0%}")
    cols[4].metric("Failure-mode pass", f"{hard['failure_mode_pass_rate']:.0%}", f"{hard['failure_mode_pass_rate']-base['failure_mode_pass_rate']:+.0%}")
    st.subheader("Baseline vs hardened")
    st.dataframe(pd.DataFrame([base, hard], index=["baseline", "hardened"]), use_container_width=True)
    st.subheader("Hardened case results")
    rows = [{k: v for k, v in r.items() if k != "retrieved"} for r in report["hardened"]["cases"]]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    st.download_button("Download JSON report", json.dumps(report, indent=2), "rag-agent-eval-report.json", "application/json")
