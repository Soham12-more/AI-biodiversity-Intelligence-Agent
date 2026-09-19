"""Knowledge layer: structured evidence cards + intervention catalog + a vector index over
cards and any documents dropped into kb/docs (PDF / txt / md).

Retrieval is hybrid: TF-IDF (always available, deterministic, CI-friendly) plus dense
sentence embeddings when `sentence-transformers` is installed. Scores are blended.
"""
from __future__ import annotations

import os
import pickle
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent.parent
KB_DIR = Path(os.getenv("KB_DIR", ROOT / "kb"))
INDEX_PATH = Path(os.getenv("INDEX_PATH", ROOT / "data" / "index.pkl"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


@dataclass
class Chunk:
    id: str            # evidence id, or "doc:<file>#<n>"
    source_id: str     # evidence id this chunk supports (doc chunks: file stem)
    text: str
    kind: str          # "card" | "doc"


@dataclass
class KB:
    evidence: dict[str, dict]
    stressors: dict[str, dict]
    interventions: list[dict]
    causal_links: list[list[str]]
    chunks: list[Chunk] = field(default_factory=list)


def _card_text(c: dict) -> str:
    return f"{c['title']}. {c['finding']} Metrics: {', '.join(c.get('metrics', []))}. ({c['authors']} {c['year']})"


def _chunk_doc(path: Path, size: int = 900, overlap: int = 150) -> list[str]:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        text = "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")
    text = re.sub(r"\s+", " ", text).strip()
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        i += size - overlap
    return [c for c in out if len(c) > 80]


@lru_cache(maxsize=1)
def load_kb() -> KB:
    ev = {c["id"]: c for c in yaml.safe_load((KB_DIR / "evidence.yaml").read_text())}
    cat = yaml.safe_load((KB_DIR / "interventions.yaml").read_text())
    # integrity: every intervention must cite existing evidence -> no orphan claims
    for iv in cat["interventions"]:
        for eid in iv["evidence"] + iv.get("caveat_evidence", []):
            if eid not in ev:
                raise ValueError(f"Intervention {iv['id']} cites unknown evidence '{eid}'")
    kb = KB(ev, cat["stressors"], cat["interventions"], cat.get("causal_links", []))
    kb.chunks = [Chunk(c["id"], c["id"], _card_text(c), "card") for c in ev.values()]
    docs = KB_DIR / "docs"
    if docs.exists():
        for p in sorted(docs.iterdir()):
            if p.suffix.lower() in {".pdf", ".txt", ".md"} and p.name.lower() != "readme.md":
                for n, t in enumerate(_chunk_doc(p)):
                    kb.chunks.append(Chunk(f"doc:{p.name}#{n}", p.stem, t, "doc"))
    return kb


class Retriever:
    def __init__(self, kb: KB):
        self.kb = kb
        texts = [c.text for c in kb.chunks]
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self.tfidf_m = self.tfidf.fit_transform(texts)
        self.dense = None
        self.encoder = None
        if os.getenv("USE_EMBEDDINGS", "auto") != "0":
            try:
                from sentence_transformers import SentenceTransformer  # optional
                self.encoder = SentenceTransformer(EMBED_MODEL)
                self.dense = self.encoder.encode(texts, normalize_embeddings=True)
            except Exception:
                self.encoder = None
        self.mode = "hybrid(tfidf+dense)" if self.dense is not None else "tfidf"

    def search(self, query: str, k: int = 5, restrict_to: set[str] | None = None) -> list[dict]:
        q = self.tfidf.transform([query])
        scores = (self.tfidf_m @ q.T).toarray().ravel()
        if self.dense is not None:
            qd = self.encoder.encode([query], normalize_embeddings=True)[0]
            scores = 0.5 * scores + 0.5 * (self.dense @ qd)
        order = np.argsort(-scores)
        out = []
        for i in order:
            ch = self.kb.chunks[i]
            if restrict_to and ch.source_id not in restrict_to:
                continue
            if scores[i] <= 0:
                break
            out.append({"chunk_id": ch.id, "source_id": ch.source_id, "kind": ch.kind,
                        "score": round(float(scores[i]), 3), "text": ch.text[:400]})
            if len(out) >= k:
                break
        return out

    def save(self, path: Path = INDEX_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"n_chunks": len(self.kb.chunks), "mode": self.mode,
                         "vocab": len(self.tfidf.vocabulary_)}, f)


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever(load_kb())
