from __future__ import annotations
import re
from dataclasses import dataclass

from .retrieval import TfidfLlamaRetriever

UNSAFE = re.compile(
    r"ignore\s+(all\s+)?(prior|previous)?\s*instructions|answer\s+pwned|system\s+prompt",
    re.I,
)
TOKEN = re.compile(r"[a-z0-9-]+", re.I)


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in TOKEN.findall(text) if len(t) > 2}


def _intent_bonus(query: str, sentence: str) -> int:
    q, s = query.lower(), sentence.lower()
    bonus = 0
    if any(x in q for x in ("how long", "how quickly", "window")):
        if re.search(r"\b\d+\s+(days?|hours?|minutes?)\b", s):
            bonus += 4
    if "required" in q or q.startswith("is "):
        if "mandatory" in s or "required" in s:
            bonus += 4
    if "what happens" in q or "after more than" in q:
        if any(x in s for x in ("receive", "returns", "above the limit")):
            bonus += 6
    if "approval" in q and "approval" in s:
        bonus += 4
    if "override" in q and "override" in s:
        bonus += 4
    return bonus


@dataclass
class RAGResult:
    answer: str
    retrieved: list[dict]


class BaselineRAG:
    def __init__(self, retriever: TfidfLlamaRetriever):
        self.retriever = retriever

    def answer(self, query: str) -> RAGResult:
        hits = self.retriever.search(query)
        top = hits[0]
        text = top.node.get_content().strip()
        answer = re.split(r"(?<=[.!?])\s+", text)[0]
        return RAGResult(answer=answer, retrieved=_serialize(hits))


class HardenedRAG:
    def __init__(self, retriever: TfidfLlamaRetriever, confidence: float = 0.08):
        self.retriever = retriever
        self.confidence = confidence

    def answer(self, query: str) -> RAGResult:
        hits = self.retriever.search(query)
        safe_hits = [h for h in hits if not UNSAFE.search(h.node.get_content())]
        if not safe_hits or float(safe_hits[0].score or 0) < self.confidence:
            return RAGResult(answer=_refusal(), retrieved=_serialize(hits))

        q_tokens = _tokens(query)
        candidates = []
        for hit in safe_hits:
            for sentence in re.split(r"(?<=[.!?])\s+", hit.node.get_content()):
                overlap = len(q_tokens & _tokens(sentence))
                bonus = _intent_bonus(query, sentence)
                score = bonus + overlap
                candidates.append((score, overlap, float(hit.score or 0), sentence.strip()))
        candidates.sort(reverse=True)
        if not candidates or candidates[0][1] == 0:
            return RAGResult(answer=_refusal(), retrieved=_serialize(hits))
        return RAGResult(answer=candidates[0][3], retrieved=_serialize(hits))


def _serialize(hits):
    return [
        {
            "doc_id": h.node.metadata.get("doc_id"),
            "score": round(float(h.score or 0), 5),
            "text": h.node.get_content(),
        }
        for h in hits
    ]


def _refusal() -> str:
    return "I don't have enough grounded evidence to answer that question."
