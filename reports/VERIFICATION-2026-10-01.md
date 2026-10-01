# Reverification — 1 October 2026

The previous environments use Anaconda Python 3.9 and fail to import SSL.
A separate `.venv-runtime` was created with Python 3.12.10, then installed from
the existing pinned requirements and editable project metadata.

- `python -m pytest -q`: 7 passed, including the Streamlit dashboard test.
- `python run_benchmark.py`: baseline 8/16; hardened 16/16 on the included
  frozen synthetic benchmark.
- LangChain's pinned older version emits Pydantic deprecation warnings;
  these are not test failures, but migration remains future maintenance work.

GitHub CI now reproduces tests and the benchmark on Python 3.12. The Windows
launcher checks the chosen runtime before starting a loopback-only dashboard.
These measurements apply only to fixture behavior; they do not establish
production accuracy or hosted-model reliability.
