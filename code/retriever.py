"""
retriever.py — Loads the local support corpus and retrieves relevant docs via TF-IDF.
"""
import os
import glob
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

COMPANY_DIRS = {
    "HackerRank": DATA_DIR / "hackerrank",
    "Claude": DATA_DIR / "claude",
    "Visa": DATA_DIR / "visa",
}


def load_corpus(company: str | None = None) -> list[dict]:
    """Load markdown/text files from the corpus. Filter by company if given."""
    docs = []
    dirs_to_load = {}

    if company and company in COMPANY_DIRS:
        dirs_to_load[company] = COMPANY_DIRS[company]
    elif company == "None" or company is None:
        dirs_to_load = COMPANY_DIRS
    else:
        dirs_to_load = COMPANY_DIRS

    for cname, cdir in dirs_to_load.items():
        if not cdir.exists():
            continue
        for fpath in glob.glob(str(cdir / "**/*.md"), recursive=True):
            try:
                text = Path(fpath).read_text(encoding="utf-8", errors="ignore")
                # Keep reasonable chunk size
                if len(text.strip()) < 20:
                    continue
                docs.append({
                    "company": cname,
                    "path": fpath,
                    "filename": os.path.basename(fpath),
                    "text": text[:4000],  # cap per doc
                })
            except Exception:
                pass
        # Also load .txt files
        for fpath in glob.glob(str(cdir / "**/*.txt"), recursive=True):
            try:
                text = Path(fpath).read_text(encoding="utf-8", errors="ignore")
                if len(text.strip()) < 20:
                    continue
                docs.append({
                    "company": cname,
                    "path": fpath,
                    "filename": os.path.basename(fpath),
                    "text": text[:4000],
                })
            except Exception:
                pass

    return docs


class CorpusRetriever:
    def __init__(self, company: str | None = None):
        self.docs = load_corpus(company)
        if not self.docs:
            self.vectorizer = None
            self.matrix = None
        else:
            self.vectorizer = TfidfVectorizer(
                stop_words="english",
                max_features=10000,
                ngram_range=(1, 2),
            )
            texts = [d["text"] for d in self.docs]
            self.matrix = self.vectorizer.fit_transform(texts)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Return top_k most relevant docs for the query."""
        if not self.docs or self.vectorizer is None:
            return []
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.matrix).flatten()
        top_idx = np.argsort(scores)[::-1][:top_k]
        results = []
        for i in top_idx:
            if scores[i] > 0.0:
                results.append({**self.docs[i], "score": float(scores[i])})
        return results

    def corpus_size(self) -> int:
        return len(self.docs)


# Singleton cache — one retriever per company to avoid reloading
_cache: dict[str, CorpusRetriever] = {}


def get_retriever(company: str | None) -> CorpusRetriever:
    key = company or "ALL"
    if key not in _cache:
        _cache[key] = CorpusRetriever(company)
    return _cache[key]