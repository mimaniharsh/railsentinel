"""Policy knowledge base with TF-IDF retrieval (scikit-learn, no external service)."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from railsentinel.config import STATIC_DATA_DIR

KB_DIR = STATIC_DATA_DIR / "knowledge"


def _parse(path: Path) -> list[dict]:
    raw = path.read_text(encoding="utf-8")
    meta: dict = {}
    body = raw
    if raw.startswith("---"):
        _, front, body = raw.split("---", 2)
        meta = yaml.safe_load(front) or {}
    chunks = []
    for section in re.split(r"\n(?=## )", body.strip()):
        lines = section.strip().splitlines()
        if not lines:
            continue
        heading = lines[0].lstrip("# ").strip() if lines[0].startswith("#") else meta.get("title", path.stem)
        text = " ".join(l.strip() for l in lines[1:] if l.strip()) if lines[0].startswith("#") else " ".join(lines)
        if text:
            chunks.append({"doc": path.stem, "title": meta.get("title", path.stem), "source": meta.get("source", ""),
                           "url": meta.get("url", ""), "heading": heading, "text": text})
    return chunks


class KnowledgeBase:
    def __init__(self, folder: Path = KB_DIR):
        self.chunks: list[dict] = []
        for path in sorted(Path(folder).glob("*.md")):
            self.chunks.extend(_parse(path))
        corpus = [f"{c['title']} {c['heading']} {c['text']}" for c in self.chunks]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self.matrix = self.vectorizer.fit_transform(corpus)

    def search(self, query: str, k: int = 3, min_score: float = 0.05) -> list[dict]:
        if not query.strip():
            return []
        sims = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        order = np.argsort(-sims)[:k]
        return [{**self.chunks[i], "score": round(float(sims[i]), 4)} for i in order if sims[i] >= min_score]

    def documents(self) -> list[dict]:
        seen = {}
        for c in self.chunks:
            seen.setdefault(c["doc"], {"doc": c["doc"], "title": c["title"], "source": c["source"], "url": c["url"],
                                       "sections": 0})
            seen[c["doc"]]["sections"] += 1
        return list(seen.values())
