"""
A lightweight in-memory implementation of the pieces used by the provided tests.

This file provides:
- LearningContent dataclass
- ContentType / DifficultyLevel enums
- UserProfile dataclass
- VectorDBManager: in-memory storage + simple TF-IDF dense search + BM25-like search
- LearnoraContentDiscovery: a simple discovery wrapper with an in-memory (or optional) cache
- compute_ndcg and compute_mrr: evaluation metrics used in tests

This implementation avoids external dependencies and is deterministic so the unit tests
you provided should run against it without network access.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Dict, Optional, Any, Tuple, Iterable, Set
from datetime import datetime
import math
import re
import threading

# ---------------------------
# Basic data structures
# ---------------------------

class ContentType(Enum):
    ARTICLE = "article"
    VIDEO = "video"
    COURSE = "course"
    NOTE = "note"

class DifficultyLevel(Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"

@dataclass
class LearningContent:
    id: str
    title: str
    content_type: ContentType
    source: str
    url: str
    description: str
    difficulty: DifficultyLevel
    duration_minutes: int
    tags: List[str]
    prerequisites: List[str]
    metadata: Dict[str, Any]
    created_at: datetime
    checksum: Optional[str] = None

@dataclass
class UserProfile:
    user_id: str
    knowledge_areas: Dict[str, str]
    learning_goals: List[str]
    preferred_formats: List[ContentType]
    available_time_daily: int
    learning_style: str

# ---------------------------
# Simple text processing utils
# ---------------------------

_token_re = re.compile(r"\w+")

def tokenize(text: str) -> List[str]:
    if text is None:
        return []
    return [t.lower() for t in _token_re.findall(text)]

def join_content_text(c: LearningContent) -> str:
    parts = [c.title or "", c.description or ""]
    if c.tags:
        parts.append(" ".join(c.tags))
    # include difficulty and content type as tokens for weak signal
    parts.append(c.difficulty.value if isinstance(c.difficulty, DifficultyLevel) else str(c.difficulty))
    parts.append(c.content_type.value if isinstance(c.content_type, ContentType) else str(c.content_type))
    return " ".join(parts)

# ---------------------------
# VectorDBManager
# ---------------------------

class VectorDBManager:
    """
    In-memory storage of LearningContent items with:
     - a simple TF-IDF style dense vector representation (sparse dict form)
     - a BM25-style ranking implementation for lexical retrieval

    Public methods used by tests:
     - add_contents(list_of_LearningContent)
     - search(query, top_k) -> list of LearningContent (dense/hybrid)
     - _bm25_search(query, top_k) -> list of (content_id, score)
    """

    def __init__(self, collection_name: str = "default_collection"):
        self.collection_name = collection_name
        self._contents: Dict[str, LearningContent] = {}
        # tf-idf like structures
        self._doc_term_freqs: Dict[str, Dict[str, float]] = {}
        self._doc_norms: Dict[str, float] = {}
        self._doc_lengths: Dict[str, int] = {}
        self._df: Dict[str, int] = {}
        self._N = 0
        # BM25 parameters
        self._k1 = 1.5
        self._b = 0.75
        # lock for thread-safety on add/rebuild
        self._lock = threading.Lock()

    def add_contents(self, contents: Iterable[LearningContent]) -> None:
        """
        Add or replace contents in the in-memory DB and rebuild indexes.
        Deterministic: if same content id exists it will be replaced.
        """
        with self._lock:
            changed = False
            for c in contents:
                if c.id not in self._contents:
                    changed = True
                else:
                    # if content changed (shallow compare) mark changed
                    old = self._contents[c.id]
                    if old.title != c.title or old.description != c.description or old.tags != c.tags:
                        changed = True
                self._contents[c.id] = c
            # Rebuild indexes whenever add_contents is called to keep things simple and deterministic
            self._rebuild_indexes()

    def _rebuild_indexes(self):
        # reset
        self._doc_term_freqs = {}
        self._doc_norms = {}
        self._doc_lengths = {}
        self._df = {}
        self._N = len(self._contents)
        # build term frequencies and document frequencies
        for cid, content in sorted(self._contents.items()):  # sorted for determinism
            text = join_content_text(content)
            tokens = tokenize(text)
            tf: Dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self._doc_term_freqs[cid] = {k: float(v) for k, v in tf.items()}
            self._doc_lengths[cid] = len(tokens)
            for term in tf.keys():
                self._df[term] = self._df.get(term, 0) + 1
        # compute document vector norms (tf-idf norm)
        for cid, freqs in self._doc_term_freqs.items():
            norm = 0.0
            for term, f in freqs.items():
                idf = self._idf(term)
                tf_idf = f * idf
                norm += tf_idf * tf_idf
            norm = math.sqrt(norm) if norm > 0 else 1.0
            self._doc_norms[cid] = norm

    def _idf(self, term: str) -> float:
        # add 1 smoothing
        df = self._df.get(term, 0)
        return math.log((self._N - df + 0.5) / (df + 0.5) + 1.0)

    def _vector_for_query(self, query: str) -> Tuple[Dict[str, float], float]:
        tokens = tokenize(query)
        tf: Dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        vec: Dict[str, float] = {}
        norm = 0.0
        for term, f in tf.items():
            idf = self._idf(term)
            val = f * idf
            vec[term] = val
            norm += val * val
        norm = math.sqrt(norm) if norm > 0 else 1.0
        return vec, norm

    def _cosine_similarity(self, qvec: Dict[str, float], qnorm: float, doc_id: str) -> float:
        doc_vec = self._doc_term_freqs.get(doc_id, {})
        # doc tf * idf
        s = 0.0
        for t, qv in qvec.items():
            dv = doc_vec.get(t)
            if dv is not None:
                s += qv * (dv * self._idf(t))
        denom = qnorm * self._doc_norms.get(doc_id, 1.0)
        return s / denom if denom != 0 else 0.0

    def _bm25_score_for_doc(self, query_terms: List[str], doc_id: str) -> float:
        score = 0.0
        freqs = self._doc_term_freqs.get(doc_id, {})
        dl = self._doc_lengths.get(doc_id, 0)
        avgdl = (sum(self._doc_lengths.values()) / self._N) if self._N > 0 else 1.0
        for term in query_terms:
            f = freqs.get(term, 0.0)
            if f <= 0:
                continue
            idf = self._idf(term)
            denom = f + self._k1 * (1 - self._b + self._b * (dl / avgdl))
            score += idf * ((f * (self._k1 + 1)) / denom)
        return float(score)

    def _bm25_search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Return list of tuples (content_id, score) sorted descending by BM25 score.
        The test calls vdb._bm25_search directly, so we expose this method.
        """
        tokens = tokenize(query)
        if not tokens:
            return []
        scores: List[Tuple[str, float]] = []
        for cid in sorted(self._contents.keys()):  # deterministic order
            sc = self._bm25_score_for_doc(tokens, cid)
            if sc > 0:
                scores.append((cid, sc))
        # sort descending by score then by id for determinism
        scores.sort(key=lambda x: (-x[1], x[0]))
        return scores[:top_k]

    # Public alias (tests use _bm25_search directly)
    _bm25_search = _bm25_search

    def search(self, query: str, top_k: int = 10) -> List[LearningContent]:
        """
        Dense/hybrid retrieval:
         - compute a TF-IDF-like vector for the query and compute cosine similarity
         - also compute BM25 scores and combine them (simple weighted sum)
         - return list of LearningContent objects sorted by final score
        """
        qvec, qnorm = self._vector_for_query(query)
        tokens = list(qvec.keys())
        results: List[Tuple[str, float]] = []
        for cid in sorted(self._contents.keys()):
            cos = self._cosine_similarity(qvec, qnorm, cid)
            bm25 = self._bm25_score_for_doc(tokens, cid)
            # combine signals: cosine normalized [0,1] approx, bm25 unbounded -> scale bm25 via sigmoid-ish
            # For simplicity, convert bm25 to a bounded score by dividing by (1+bm25) to keep magnitude reasonable.
            bm25_scaled = bm25 / (1.0 + bm25) if bm25 > 0 else 0.0
            final = 0.6 * cos + 0.4 * bm25_scaled
            if final > 0:
                results.append((cid, final))
        # sort by score desc then id for determinism
        results.sort(key=lambda x: (-x[1], x[0]))
        results = results[:top_k]
        return [self._contents[cid] for cid, _ in results]

# ---------------------------
# LearnoraContentDiscovery
# ---------------------------

class LearnoraContentDiscovery:
    """
    Small wrapper that orchestrates a VectorDBManager and provides caching.
    If redis_url is None, an in-memory cache is used.
    discover_and_personalize returns a dict with results and stats.
    """

    def __init__(self, openai_api_key: Optional[str] = None, redis_url: Optional[str] = None):
        # openai_api_key present but unused in these tests (we don't call the API)
        self.openai_api_key = openai_api_key
        self.redis_url = redis_url
        self.vector_db = VectorDBManager(collection_name="learnora_contents")
        # in-memory cache used when redis_url is None
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_lock = threading.Lock()

    def _cache_key(self, query: str, user_profile: UserProfile) -> str:
        # deterministic key built from query and user id and normalized preferred formats
        pf = ",".join(sorted([p.value for p in user_profile.preferred_formats])) if user_profile.preferred_formats else ""
        key = f"q={{query}}|uid={{user_profile.user_id}}|pf={{pf}}|goals={{'|'.join(sorted(user_profile.learning_goals))}}"
        return key

    def discover_and_personalize(self, query: str, user_profile: UserProfile, refresh_content: bool = False) -> Dict[str, Any]:
        """
        Main entry point used in tests. If refresh_content=False we do not attempt
        to fetch any external content. We search the local vector DB and produce
        a deterministic result with 'stats' included.
        Caching: if redis_url is None use in-memory cache.
        """
        key = self._cache_key(query, user_profile)
        # check cache first
        if not refresh_content:
            with self._cache_lock:
                if key in self._memory_cache:
                    return self._memory_cache[key]

        # build results from vector DB
        hits = self.vector_db.search(query, top_k=10)
        # simple personalization: prefer matching preferred content types and difficulty close to user knowledge level if present
        preferred_types: Set[ContentType] = set(user_profile.preferred_formats or [])
        personalized: List[Dict[str, Any]] = []
        for c in hits:
            score = 1.0
            if preferred_types and c.content_type not in preferred_types:
                score -= 0.2
            # boost beginner for beginner users
            if user_profile.knowledge_areas:
                # take any listed level (simple heuristic)
                try:
                    # pick first knowledge area level
                    first_level = next(iter(user_profile.knowledge_areas.values()))
                    if first_level and first_level.lower() == "beginner" and c.difficulty == DifficultyLevel.BEGINNER:
                        score += 0.1
                except StopIteration:
                    pass
            personalized.append({
                "id": c.id,
                "title": c.title,
                "url": c.url,
                "score": round(score, 0) if isinstance(score, float) and score.is_integer() else round(score, 4),
                "difficulty": c.difficulty.value if isinstance(c.difficulty, DifficultyLevel) else str(c.difficulty),
                "content_type": c.content_type.value if isinstance(c.content_type, ContentType) else str(c.content_type),
            })

        # sort personalized results by score then by id for determinism
        personalized.sort(key=lambda x: (-x["score"], x["id"]))

        out = {
            "query": query,
            "user_id": user_profile.user_id,
            "results": personalized,
            "stats": {
                "num_hits": len(hits),
                "returned": len(personalized)
            },
            # attach deterministic representation to make equality checks in tests reliable
            "cached": False
        }

        # store in cache if not refreshing
        if not refresh_content:
            with self._cache_lock:
                # store the dict itself (not a copy) to make second call return an identical object for equality
                self._memory_cache[key] = out

        return out

# ---------------------------
# Metrics
# ---------------------------

def compute_dcg(relevances: List[float], k: int) -> float:
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        denom = math.log2(i + 2)  # positions are 1-based; i=0 -> log2(2)
        dcg += rel / denom
    return dcg

def compute_ndcg(predictions: List[str], ground_truth: Dict[str, float], k: int = 10) -> float:
    """
    predictions: list of item ids in ranked order
    ground_truth: mapping id -> graded relevance (float). Items not present have relevance 0.
    k: cutoff
    """
    # relevance in the order of predictions
    rels = [float(ground_truth.get(pid, 0.0)) for pid in predictions[:k]]
    dcg = compute_dcg(rels, k)
    # ideal DCG: sort ground-truth relevances descending
    ideal_rels = sorted([v for v in ground_truth.values() if v is not None], reverse=True)
    idcg = compute_dcg(ideal_rels, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg

def compute_mrr(predictions: List[str], ground_truth_set: Set[str]) -> float:
    """
    predictions: list of item ids in ranked order
    ground_truth_set: set of relevant item ids
    MRR = 1 / rank_of_first_relevant
    """
    for i, pid in enumerate(predictions, start=1):
        if pid in ground_truth_set:
            return 1.0 / float(i)
    return 0.0

# Expose public names
__all__ = [
    "LearningContent", "ContentType", "DifficultyLevel",
    "VectorDBManager", "LearnoraContentDiscovery", "compute_ndcg", "compute_mrr",
    "UserProfile"
]