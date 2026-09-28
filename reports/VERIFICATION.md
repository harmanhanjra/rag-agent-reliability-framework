# Verification Report — 2026-09-28

## Environment
- Python 3.9.12
- LlamaIndex Core 0.10.68.post1
- LangChain 0.2.17
- LangChain Core 0.2.43
- Pydantic 2.10.6
- scikit-learn 1.6.1
- Streamlit 1.50.0
- pytest 8.4.2

## Frozen benchmark
The deterministic fixture contains 16 cases: 10 RAG cases and 6 tool/agent cases.

### Baseline
- Task completion: 50.00%
- RAG answer accuracy: 50.00%
- Retrieval hit-rate@3: 100.00%
- MRR: 92.59%
- Groundedness: 80.00%
- Tool reliability: 50.00%
- Failure-mode pass rate: 16.67%
- Passed: 8/16

### Hardened
- Task completion: 100.00%
- RAG answer accuracy: 100.00%
- Retrieval hit-rate@3: 100.00%
- MRR: 92.59%
- Groundedness: 100.00%
- Tool reliability: 100.00%
- Failure-mode pass rate: 100.00%
- Passed: 16/16

## Automated verification
pytest result: 7 passed.

The test suite covers LlamaIndex retrieval, poisoned-context handling, unanswerable-query refusal, LangChain tool schema validation, destructive-action approval gating, the 95%+ task-completion threshold, and a headless Streamlit dashboard run.

## Local web verification
Streamlit health endpoint returned HTTP 200 with body "ok".
The root page returned HTTP 200.

## Scope
These results apply only to the included synthetic deterministic benchmark. They are reproducible evidence for this fixture suite, not a claim of 100% performance on arbitrary production traffic.
