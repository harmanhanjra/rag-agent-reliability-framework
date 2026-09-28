from __future__ import annotations
import json
from pathlib import Path
from typing import Any

from .rag import BaselineRAG, HardenedRAG, UNSAFE
from .retrieval import TfidfLlamaRetriever
from .tools import BaselineToolAgent, HardenedToolAgent

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def _load(name: str):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def _contains_terms(text: str, terms: list[str]) -> bool:
    lower = text.lower()
    return all(term.lower() in lower for term in terms)


def _rag_case(case: dict, pipeline) -> dict:
    result = pipeline.answer(case["query"])
    retrieved_ids = [x["doc_id"] for x in result.retrieved]
    if case.get("expect_refusal"):
        correct = result.answer.startswith("I don't have enough")
    else:
        correct = _contains_terms(result.answer, case.get("expected_terms", []))
    rank = None
    if case.get("expected_doc") in retrieved_ids:
        rank = retrieved_ids.index(case["expected_doc"]) + 1
    safe_context = " ".join(
        x["text"] for x in result.retrieved if not UNSAFE.search(x["text"])
    )
    grounded = result.answer.startswith("I don't have enough") or (
        result.answer.lower() in safe_context.lower()
    )
    return {
        "id": case["id"], "kind": "rag",
        "failure_mode": case.get("failure_mode"),
        "passed": bool(correct and grounded), "answer": result.answer,
        "retrieval_evaluable": bool(case.get("expected_doc")),
        "retrieval_rank": rank, "hit_at_3": rank is not None and rank <= 3,
        "grounded": grounded, "retrieved": result.retrieved,
    }


def _tool_case(case: dict, agent) -> dict:
    result = agent.run(case["query"])
    tool_ok = case.get("expected_tool") is None or (
        result.tool == case.get("expected_tool")
    )
    passed = tool_ok and result.status == case["expected_status"]
    return {
        "id": case["id"], "kind": "tool",
        "failure_mode": case.get("failure_mode"),
        "passed": bool(passed), "tool": result.tool, "args": result.args,
        "status": result.status, "output": result.output,
    }


def _aggregate(rows: list[dict]) -> dict[str, Any]:
    rag = [r for r in rows if r["kind"] == "rag"]
    tools = [r for r in rows if r["kind"] == "tool"]
    failures = [r for r in rows if r.get("failure_mode")]
    retrieval = [r for r in rag if r.get("retrieval_evaluable")]
    ranks = [r["retrieval_rank"] for r in retrieval if r.get("retrieval_rank")]
    return {
        "task_completion_rate": round(sum(r["passed"] for r in rows) / len(rows), 4),
        "rag_answer_accuracy": round(sum(r["passed"] for r in rag) / len(rag), 4),
        "retrieval_hit_rate_at_3": round(sum(r["hit_at_3"] for r in retrieval) / len(retrieval), 4),
        "mrr": round(sum(1 / r for r in ranks) / len(retrieval), 4),
        "groundedness_rate": round(sum(r["grounded"] for r in rag) / len(rag), 4),
        "tool_reliability": round(sum(r["passed"] for r in tools) / len(tools), 4),
        "failure_mode_pass_rate": round(sum(r["passed"] for r in failures) / len(failures), 4),
        "total_cases": len(rows), "passed_cases": sum(r["passed"] for r in rows),
    }


def _run_variant(name: str, rag_pipeline, tool_agent, suite: dict) -> dict:
    rows = [_rag_case(c, rag_pipeline) for c in suite["rag_cases"]]
    rows += [_tool_case(c, tool_agent) for c in suite["tool_cases"]]
    return {"variant": name, "metrics": _aggregate(rows), "cases": rows}


def run_benchmark() -> dict:
    knowledge, suite = _load("knowledge.json"), _load("benchmark.json")
    retriever = TfidfLlamaRetriever(knowledge, top_k=3)
    baseline = _run_variant(
        "baseline", BaselineRAG(retriever), BaselineToolAgent(), suite
    )
    hardened = _run_variant(
        "hardened", HardenedRAG(retriever), HardenedToolAgent(), suite
    )
    return {"baseline": baseline, "hardened": hardened}
