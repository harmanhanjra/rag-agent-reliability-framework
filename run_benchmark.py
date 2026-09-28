import json
from pathlib import Path

from rag_eval import run_benchmark

report = run_benchmark()
Path("reports").mkdir(exist_ok=True)
Path("reports/latest.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)
print(json.dumps({
    "baseline": report["baseline"]["metrics"],
    "hardened": report["hardened"]["metrics"],
}, indent=2))
