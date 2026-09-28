from __future__ import annotations
from typing import Iterable, List

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfLlamaRetriever(BaseRetriever):
    """Offline deterministic retriever using LlamaIndex nodes and retriever contracts."""

    def __init__(self, records: Iterable[dict], top_k: int = 3):
        super().__init__()
        documents = [
            Document(text=r["text"], metadata={"doc_id": r["id"]})
            for r in records
        ]
        splitter = SentenceSplitter(chunk_size=128, chunk_overlap=16)
        self.nodes = splitter.get_nodes_from_documents(documents)
        self.top_k = top_k
        self.vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), sublinear_tf=True
        )
        self.matrix = self.vectorizer.fit_transform([n.get_content() for n in self.nodes])

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query_vec = self.vectorizer.transform([query_bundle.query_str])
        scores = cosine_similarity(query_vec, self.matrix).ravel()
        order = scores.argsort()[::-1][: self.top_k]
        return [
            NodeWithScore(node=self.nodes[i], score=float(scores[i]))
            for i in order
        ]

    def search(self, query: str) -> List[NodeWithScore]:
        return self.retrieve(query)
