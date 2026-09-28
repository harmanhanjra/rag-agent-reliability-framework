# RAG Evaluation & Agent Reliability Framework

A reproducible evaluation harness for RAG and tool-using agents built with LlamaIndex and LangChain.

This repository makes reliability claims measurable. It compares a naive baseline against a hardened pipeline on the same deterministic benchmark instead of hard-coding a success percentage.

## What it measures

- Retrieval hit-rate@3
- Mean reciprocal rank (MRR)
- Groundedness / evidence support
- Tool-selection and argument reliability
- Task-completion rate
- Failure-mode coverage

## Technology

- LlamaIndex: documents, sentence chunking, retriever abstraction, scored nodes and query contracts.
- LangChain: StructuredTool definitions plus Pydantic-backed tool argument schemas and invocation.
- scikit-learn: deterministic TF-IDF scoring so the benchmark works without paid API keys.
- Streamlit: interactive dashboard and JSON report export.
- pytest: automated regression and reliability checks.

## Architecture

The corpus is converted into LlamaIndex Document objects and split into nodes. A deterministic TF-IDF retriever implements LlamaIndex BaseRetriever and returns NodeWithScore results.

Two RAG variants are evaluated:
1. BaselineRAG blindly uses the highest-ranked retrieved content.
2. HardenedRAG rejects injected instructions, applies a retrieval-confidence threshold, selects relevant evidence, and refuses unsupported questions.

Tool evaluation uses real LangChain StructuredTool objects:
1. The baseline executes crude tool plans and can cross a destructive boundary.
2. The hardened agent validates arguments through each tool's Pydantic schema, uses an allowlist, refuses unknown tools, and requires approval for destructive actions.

## Failure modes in the fixture suite

- Unanswerable questions
- Poisoned / prompt-injected retrieval context
- Destructive tool requests
- Missing required arguments
- Unknown tools
- Out-of-range tool arguments

## Run locally

Python 3.9+ is supported by the pinned dependency set.

Create a virtual environment, install the project plus requirements, run run_benchmark.py, then run pytest. Optionally start the Streamlit dashboard with app.py.

The benchmark writes its full evidence report to reports/latest.json.

## Interpreting the score

The included benchmark is synthetic and deterministic. A score of 95% or higher on this fixture suite only proves performance on these checked cases. It does not imply 95% accuracy on arbitrary production workloads.

For a production evaluation, replace or extend data/benchmark.json with representative user traces, domain questions, adversarial cases and tool failures. Keep the same success criteria and compare revisions against a frozen evaluation set.

## Resume claim

The project is designed to support the statement that RAG accuracy, tool reliability, hallucination behavior and failure-mode coverage were benchmarked. Any percentage used on a resume should come from an actual saved benchmark report.

## Project layout

- src/rag_eval/retrieval.py — LlamaIndex retrieval implementation
- src/rag_eval/rag.py — baseline and hardened RAG behavior
- src/rag_eval/tools.py — LangChain structured tools and safety layer
- src/rag_eval/benchmark.py — evaluation runner and metrics
- data/knowledge.json — controlled knowledge corpus
- data/benchmark.json — frozen benchmark cases
- tests/test_framework.py — automated verification
- app.py — Streamlit dashboard
- reports/latest.json — latest generated evidence report

## Limitations

The offline benchmark avoids an external LLM so results are reproducible and free to run. It tests orchestration, retrieval, grounding, tool routing, schemas and failure handling rather than the linguistic quality of a hosted model.

A future provider adapter can add OpenAI, Anthropic, Gemini or NVIDIA NIM while preserving this deterministic suite as a regression baseline.
