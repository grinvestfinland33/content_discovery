"""Lightweight in-memory content discovery toolkit.

The module provides simple data classes that model learning content and user
profiles together with an in-memory vector database that supports lexical and
hybrid retrieval strategies.  It deliberately avoids third-party dependencies
so the code can run in restricted execution environments such as this kata.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import math
import string

@dataclass
class LearningContent:
    """Represents an item that can be recommended to a learner."""

    id: str
    title: str
    content_type: str
    source: str
    url: str
    description: str
    difficulty: str
    duration_minutes: int
    tags: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    checksum: Optional[str] = None

    def document_text(self) -> str:
        """Concatenate searchable fields into a single string."""

        parts: List[str] = [self.title, self.description]
        parts.extend(self.tags)
        parts.extend(self.prerequisites)
        for key, value in self.metadata.items():
            parts.append(f"{key}: {value}")
        return " ".join(part for part in parts if part)

@dataclass
class UserProfile:
    """Small representation of a learner used for personalization."""

    user_id: str
    knowledge_areas: Dict[str, str] = field(default_factory=dict)
    learning_goals: List[str] = field(default_factory=list)
    preferred_formats: List[str] = field(default_factory=list)
    available_time_daily: int = 60
    learning_style: str = "balanced"


class VectorDBManager:
    """In-memory index that supports BM25, dense and hybrid search."""

    def __init__(self) -> None:
        self._contents: Dict[str, LearningContent] = {}
        self._tokenized_docs: Dict[str, List[str]] = {}
        self._tfidf_vectors: Dict[str, Dict[str, float]] = {}
        self._vector_norms: Dict[str, float] = {}
        self._doc_freq: Dict[str, int] = {}
        self._doc_lengths: Dict[str, int] = {}
        self._avg_doc_len: float = 0.0

    def add_contents(self, contents: Iterable[LearningContent]) -> None:
        """Add or replace a batch of contents and rebuild indices."""

        updated = False
        for content in contents:
            updated = True
            self._contents[content.id] = content
        if updated:
            self._rebuild_indices()

    @property
    def contents(self) -> Dict[str, LearningContent]:
        return self._contents

    def search(
        self,
        query: str,
        top_k: int = 10,
        strategy: str = "hybrid",
        *,
        dense_weight: float = 0.65,
    ) -> List[Tuple[LearningContent, float]]:
        """Return ranked results for *query* using the desired strategy.

        Args:
            query: The free form search string.
            top_k: Maximum number of results to return.
            strategy: One of ``"dense"``, ``"bm25"`` or ``"hybrid"``.
            dense_weight: Combination weight used for the hybrid mode.
        """

        if not query.strip():
            return []

        if strategy not in {"dense", "bm25", "hybrid"}:
            raise ValueError(f"Unsupported strategy '{strategy}'.")

        bm25_scores = self._bm25_scores(query)
        dense_scores = self._dense_scores(query)

        if strategy == "bm25":
            combined = bm25_scores
        elif strategy == "dense":
            combined = dense_scores
        else:
            combined = self._combine_scores(bm25_scores, dense_scores, dense_weight)

        ordered_ids = sorted(combined, key=combined.get, reverse=True)
        results: List[Tuple[LearningContent, float]] = []
        for content_id in ordered_ids[:top_k]:
            results.append((self._contents[content_id], combined[content_id]))
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        translator = str.maketrans({c: " " for c in string.punctuation})
        clean = text.translate(translator).lower()
        return [token for token in clean.split() if token]

    def _rebuild_indices(self) -> None:
        self._tokenized_docs.clear()
        self._tfidf_vectors.clear()
        self._vector_norms.clear()
        self._doc_freq.clear()
        self._doc_lengths.clear()

        for content_id, content in self._contents.items():
            tokens = self._tokenize(content.document_text())
            self._tokenized_docs[content_id] = tokens
            token_counts: Dict[str, int] = {}
            for token in tokens:
                self._doc_freq[token] = self._doc_freq.get(token, 0) + 1
                token_counts[token] = token_counts.get(token, 0) + 1
            self._doc_lengths[content_id] = len(tokens)
            self._tfidf_vectors[content_id] = token_counts

        total_docs = len(self._contents)
        if total_docs == 0:
            self._avg_doc_len = 0.0
            return

        self._avg_doc_len = sum(self._doc_lengths.values()) / total_docs

        for content_id, token_counts in self._tfidf_vectors.items():
            tfidf_vector: Dict[str, float] = {}
            norm = 0.0
            for token, count in token_counts.items():
                tf = 1 + math.log(count)
                df = self._doc_freq[token]
                idf = math.log((total_docs + 1) / (df + 1)) + 1
                value = tf * idf
                tfidf_vector[token] = value
                norm += value * value
            self._tfidf_vectors[content_id] = tfidf_vector
            self._vector_norms[content_id] = math.sqrt(norm) if norm else 0.0

    def _dense_scores(self, query: str) -> Dict[str, float]:
        tokens = self._tokenize(query)
        if not tokens:
            return {}
        query_counts: Dict[str, int] = {}
        for token in tokens:
            query_counts[token] = query_counts.get(token, 0) + 1

        total_docs = len(self._contents)
        query_vector: Dict[str, float] = {}
        norm = 0.0
        for token, count in query_counts.items():
            tf = 1 + math.log(count)
            df = self._doc_freq.get(token)
            if not df:
                continue
            idf = math.log((total_docs + 1) / (df + 1)) + 1
            value = tf * idf
            query_vector[token] = value
            norm += value * value
        query_norm = math.sqrt(norm) if norm else 0.0

        scores: Dict[str, float] = {}
        if not query_vector:
            return scores
        for content_id, doc_vector in self._tfidf_vectors.items():
            numerator = 0.0
            for token, weight in query_vector.items():
                numerator += weight * doc_vector.get(token, 0.0)
            doc_norm = self._vector_norms.get(content_id, 0.0)
            if numerator and doc_norm and query_norm:
                scores[content_id] = numerator / (doc_norm * query_norm)
        return scores

    def _bm25_scores(
        self,
        query: str,
        *,
        k1: float = 1.6,
        b: float = 0.75,
    ) -> Dict[str, float]:
        tokens = self._tokenize(query)
        scores: Dict[str, float] = {}
        if not tokens:
            return scores
        total_docs = len(self._contents)
        for token in tokens:
            df = self._doc_freq.get(token)
            if not df:
                continue
            idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            for content_id, doc_tokens in self._tokenized_docs.items():
                freq = doc_tokens.count(token)
                if not freq:
                    continue
                numerator = freq * (k1 + 1)
                denominator = freq + k1 * (1 - b + b * self._doc_lengths[content_id] / (self._avg_doc_len or 1))
                scores[content_id] = scores.get(content_id, 0.0) + idf * (numerator / denominator)
        return scores

    @staticmethod
    def _combine_scores(
        bm25_scores: Dict[str, float],
        dense_scores: Dict[str, float],
        dense_weight: float,
    ) -> Dict[str, float]:
        combined: Dict[str, float] = {}
        for content_id, score in bm25_scores.items():
            combined[content_id] = combined.get(content_id, 0.0) + (1 - dense_weight) * score
        for content_id, score in dense_scores.items():
            combined[content_id] = combined.get(content_id, 0.0) + dense_weight * score
        return combined

    def _bm25_search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        scores = self._bm25_scores(query)
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return ordered[:top_k]


class LearnoraContentDiscovery:
    """Thin wrapper that combines search with simple personalization."""

    def __init__(self,
        *,
        vector_db: Optional[VectorDBManager] = None,
        openai_api_key: Optional[str] = None,
        redis_url: Optional[str] = None,
    ) -> None:
        self.vector_db = vector_db or VectorDBManager()
        self.openai_api_key = openai_api_key
        self.redis_url = redis_url
        self._cache: Dict[str, Dict[str, Any]] = {}

    def _cache_key(self, query: str, user_profile: UserProfile, strategy: str) -> str:
        profile_dict = asdict(user_profile)
        return f"{query}|{strategy}|{sorted(profile_dict.items())}"

    def discover_and_personalize(
        self,
        query: str,
        user_profile: UserProfile,
        *,
        strategy: str = "hybrid",
        top_k: int = 5,
        refresh_content: bool = False,
    ) -> Dict[str, Any]:
        if not refresh_content:
            cached = self._cache.get(self._cache_key(query, user_profile, strategy))
            if cached is not None:
                return cached

        ranked = self.vector_db.search(query, top_k=top_k, strategy=strategy)
        personalized = self._personalize_results(ranked, user_profile)
        payload = {
            "query": query,
            "user_id": user_profile.user_id,
            "strategy": strategy,
            "results": personalized,
            "stats": {
                "total_indexed": len(self.vector_db.contents),
                "returned": len(personalized),
            },
        }
        self._cache[self._cache_key(query, user_profile, strategy)] = payload
        return payload

    def _personalize_results(
        self,
        ranked_results: Sequence[Tuple[LearningContent, float]],
        user_profile: UserProfile,
    ) -> List[Dict[str, Any]]:
        if not ranked_results:
            return []

        boost_formats = {fmt.lower() for fmt in user_profile.preferred_formats}
        available_time = user_profile.available_time_daily

        adjusted: List[Tuple[LearningContent, float]] = []
        for content, score in ranked_results:
            adjusted_score = score
            if boost_formats and content.content_type.lower() in boost_formats:
                adjusted_score *= 1.1
            if available_time and content.duration_minutes <= available_time:
                adjusted_score *= 1.05
            adjusted.append((content, adjusted_score))

        adjusted.sort(key=lambda item: item[1], reverse=True)
        result_payload: List[Dict[str, Any]] = []
        for content, score in adjusted:
            result_payload.append(
                {
                    "id": content.id,
                    "title": content.title,
                    "score": round(score, 6),
                    "url": content.url,
                    "description": content.description,
                    "content_type": content.content_type,
                    "difficulty": content.difficulty,
                    "duration_minutes": content.duration_minutes,
                    "tags": content.tags,
                }
            )
        return result_payload


def compute_ndcg(predictions: List[str], ground_truth: Dict[str, float], k: int = 10) -> float:
    """Compute the normalized discounted cumulative gain (nDCG)."""

    if not predictions or not ground_truth:
        return 0.0
    dcg = 0.0
    for rank, content_id in enumerate(predictions[:k], start=1):
        rel = ground_truth.get(content_id)
        if rel is None:
            continue
        dcg += (2 ** rel - 1) / math.log2(rank + 1)

    ideal_relevances = sorted(ground_truth.values(), reverse=True)
    ideal_dcg = 0.0
    for rank, rel in enumerate(ideal_relevances[:k], start=1):
        ideal_dcg += (2 ** rel - 1) / math.log2(rank + 1)
    return dcg / ideal_dcg if ideal_dcg else 0.0


def compute_mrr(predictions: List[str], ground_truth_set: Sequence[str]) -> float:
    """Compute the mean reciprocal rank (MRR)."""

    if not predictions or not ground_truth_set:
        return 0.0
    ground_truth = set(ground_truth_set)
    for rank, content_id in enumerate(predictions, start=1):
        if content_id in ground_truth:
            return 1.0 / rank
    return 0.0


@lru_cache(maxsize=None)
def load_demo_contents() -> List[LearningContent]:
    """Provide a deterministic list of demo items for unit tests or demos."""

    now = datetime(2024, 1, 1)
    return [
        LearningContent(
            id="python-intro",
            title="Introduction to Python",
            content_type="article",
            source="demo",
            url="https://example.com/python-intro",
            description="Start writing Python programs with hands-on examples.",
            difficulty="beginner",
            duration_minutes=20,
            tags=["python", "programming", "basics"],
            created_at=now,
        ),
        LearningContent(
            id="python-advanced",
            title="Advanced Python Patterns",
            content_type="video",
            source="demo",
            url="https://example.com/python-advanced",
            description="Master decorators, context managers, and metaclasses.",
            difficulty="advanced",
            duration_minutes=45,
            tags=["python", "advanced", "patterns"],
            created_at=now,
        ),
        LearningContent(
            id="ml-fundamentals",
            title="Machine Learning Fundamentals",
            content_type="course",
            source="demo",
            url="https://example.com/ml-fundamentals",
            description="Learn supervised and unsupervised algorithms from scratch.",
            difficulty="intermediate",
            duration_minutes=90,
            tags=["machine learning", "supervised", "unsupervised"],
            created_at=now,
        ),
    ]


__all__ = [
    "LearningContent",
    "UserProfile",
    "VectorDBManager",
    "LearnoraContentDiscovery",
    "compute_ndcg",
    "compute_mrr",
    "load_demo_contents",
]