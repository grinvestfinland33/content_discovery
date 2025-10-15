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
