# Content Discovery — Lightweight In-Memory Implementation

This repository contains a lightweight, self-contained implementation of a small content-discovery toolkit intended for unit tests and local development. It intentionally avoids external dependencies and network calls so you can run deterministic tests offline.

The key file implemented for the tests is `ProjectV2_final.py`, which provides:
- Data models for learning content and user profiles
- A simple in-memory "vector DB" that supports TF‑IDF-like dense retrieval and a BM25-like lexical ranking
- A small content-discovery wrapper with an in-memory cache fallback
- Two evaluation metrics: nDCG and MRR

The included `test_content_discovery.py` (shown separately) exercises the core behaviors: BM25 search, dense/hybrid retrieval, caching, and metrics.

What I did
- Wrote a focused README that documents the code, how to run the unit tests, and how to use the main API provided by `ProjectV2_final.py`.
- Described configuration options, caching behavior, and example usage and outputs.

What's next
- If you want, I can:
  - Add more example content and demonstration scripts
  - Add a small CLI to interactively query the vector DB
  - Add optional Redis caching support or adapt the code to persist the index to disk

---

## Quick start

Requirements
- Python 3.8+ (the code uses standard library only)

Installation
- No installation required. Just place `ProjectV2_final.py` and `test_content_discovery.py` in the same folder.

Run the unit tests
```bash
python test_content_discovery.py
```

You should see output from three tests:
- BM25 + Dense pipeline
- In-memory cache fallback in `LearnoraContentDiscovery`
- nDCG and MRR metric checks

If assertions pass, the script ends with:
```
All tests finished. If assertions passed, basic consistency checks succeeded.
```

Environment variables (optional)
- `OPENAI_API_KEY`: Not used by the test implementation, but `LearnoraContentDiscovery` accepts it. A dummy key is fine for tests.
- `REDIS_URL`: If set and if you extend the class to use Redis, it would be used; by default the implementation falls back to an in-memory cache when `redis_url` is `None`.

---

## Files

- `ProjectV2_final.py` — main implementation (self-contained)
  - Data classes:
    - `LearningContent`
    - `UserProfile`
  - Enums:
    - `ContentType`
    - `DifficultyLevel`
  - Core classes:
    - `VectorDBManager`
      - add_contents(contents)
      - search(query, top_k)
      - _bm25_search(query, top_k) — exposed for tests
    - `LearnoraContentDiscovery`
      - discover_and_personalize(query, user_profile, refresh_content=False)
  - Metrics:
    - `compute_ndcg(predictions, ground_truth, k=10)`
    - `compute_mrr(predictions, ground_truth_set)`

- `test_content_discovery.py` — smoke tests that exercise the above components

---

## API reference (summary)

LearningContent (dataclass)
- id: str
- title: str
- content_type: ContentType
- source: str
- url: str
- description: str
- difficulty: DifficultyLevel
- duration_minutes: int
- tags: List[str]
- prerequisites: List[str]
- metadata: Dict[str, Any]
- created_at: datetime
- checksum: Optional[str]

UserProfile (dataclass)
- user_id: str
- knowledge_areas: Dict[str, str]
- learning_goals: List[str]
- preferred_formats: List[ContentType]
- available_time_daily: int
- learning_style: str

VectorDBManager
- add_contents(contents: Iterable[LearningContent]) -> None
  - Adds or replaces items in the in-memory DB and rebuilds TF-IDF / BM25-like indexes.
- search(query: str, top_k: int = 10) -> List[LearningContent]
  - Returns ranked results using a combined TF-IDF-like cosine similarity and BM25-like score.
- _bm25_search(query: str, top_k: int = 10) -> List[Tuple[str, float]]
  - Exposed for testing; returns (content_id, score) pairs.

LearnoraContentDiscovery
- __init__(openai_api_key: Optional[str] = None, redis_url: Optional[str] = None)
  - Uses an in-memory cache when `redis_url` is None.
- discover_and_personalize(query: str, user_profile: UserProfile, refresh_content: bool = False) -> Dict[str, Any]
  - Searches the `VectorDBManager` and returns a deterministic personalized result dict with `results` and `stats`.
  - Caches the response in-memory for identical queries & user profiles unless `refresh_content=True`.

Metrics
- compute_ndcg(predictions: List[str], ground_truth: Dict[str, float], k: int = 10) -> float
- compute_mrr(predictions: List[str], ground_truth_set: Set[str]) -> float

---

## Example usage

Simple script that adds content, performs a search, and computes metrics:

```python
from ProjectV2_final import (
    LearningContent, ContentType, DifficultyLevel,
    VectorDBManager, LearnoraContentDiscovery,
    compute_ndcg, compute_mrr, UserProfile
)
from datetime import datetime

# create content and add to DB
c = LearningContent(
    id="c1",
    title="Intro to transformers",
    content_type=ContentType.ARTICLE,
    source="unit-test",
    url="https://example.org/c1",
    description="A primer on transformers",
    difficulty=DifficultyLevel.BEGINNER,
    duration_minutes=10,
    tags=["transformers", "attention"],
    prerequisites=[],
    metadata={},
    created_at=datetime.now()
)

vdb = VectorDBManager()
vdb.add_contents([c])

# search
results = vdb.search("transformers attention")
print([r.id for r in results])

# discovery + personalization
profile = UserProfile(
    user_id="u1",
    knowledge_areas={"ML": "beginner"},
    learning_goals=["learn transformers"],
    preferred_formats=[ContentType.ARTICLE],
    available_time_daily=20,
    learning_style="reading"
)

sys = LearnoraContentDiscovery(openai_api_key="dummy", redis_url=None)
sys.vector_db.add_contents([c])
out = sys.discover_and_personalize(query="transformers", user_profile=profile)
print(out)
```

---

## Testing notes & determinism

- Indexing and iteration over documents uses deterministic ordering (sorted by content id) so tests behave consistently across runs.
- The in-memory cache stores the exact result object so repeated requests return identical objects (useful for equality checks in unit tests).
- The BM25 implementation is a simplified BM25-like scoring function tuned for deterministic unit tests (not intended as a production-grade search engine).

---

## Extending the project

Ideas for next steps:
- Persist the index using a lightweight on-disk format (JSON or SQLite).
- Replace the simple TF‑IDF vectors with embeddings (e.g., using sentence transformers) and swap in a vector index like FAISS, Chroma, or Milvus.
- Add optional Redis-backed caching (the `redis_url` field is present for this extension).
- Add more comprehensive logging and test coverage.

---

## License

This code is provided as-is for educational and testing purposes. Adapt or relicense as needed for your project.

---
```
