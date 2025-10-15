import unittest

from ProjectV2_final import (
    LearnoraContentDiscovery,
    LearningContent,
    UserProfile,
    VectorDBManager,
    compute_mrr,
    compute_ndcg,
    load_demo_contents,
)


class VectorDBManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = VectorDBManager()
        self.manager.add_contents(load_demo_contents())

    def test_bm25_search_returns_relevant_results(self) -> None:
        results = self.manager.search("python programming", strategy="bm25")
        self.assertGreater(len(results), 0)
        top_ids = [content.id for content, _ in results]
        self.assertIn("python-intro", top_ids)

    def test_dense_search_handles_unknown_terms(self) -> None:
        results = self.manager.search("quantum machine learning", strategy="dense")
        self.assertGreater(len(results), 0)
        top_ids = [content.id for content, _ in results]
        self.assertIn("ml-fundamentals", top_ids)

    def test_hybrid_search_combines_scores(self) -> None:
        bm25 = dict(self.manager._bm25_search("advanced python"))
        hybrid = self.manager.search("advanced python", strategy="hybrid")
        self.assertTrue(hybrid)
        hybrid_ids = [content.id for content, _ in hybrid]
        self.assertIn("python-advanced", hybrid_ids)
        self.assertGreater(hybrid[0][1], bm25.get(hybrid[0][0].id, 0.0))


class DiscoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        manager = VectorDBManager()
        manager.add_contents(load_demo_contents())
        self.discovery = LearnoraContentDiscovery(vector_db=manager)
        self.profile = UserProfile(
            user_id="user-1",
            preferred_formats=["video"],
            available_time_daily=50,
        )

    def test_personalization_boosts_preferred_format(self) -> None:
        payload = self.discovery.discover_and_personalize("advanced python", self.profile)
        self.assertTrue(payload["results"])
        top_result = payload["results"][0]
        self.assertEqual(top_result["content_type"], "video")

    def test_cache_returns_same_payload(self) -> None:
        first = self.discovery.discover_and_personalize("python", self.profile)
        second = self.discovery.discover_and_personalize("python", self.profile)
        self.assertIs(first, second)


class MetricsTests(unittest.TestCase):
    def test_ndcg_and_mrr(self) -> None:
        predictions = ["python-intro", "ml-fundamentals", "python-advanced"]
        ground_truth = {"python-intro": 3.0, "python-advanced": 2.0}
        ndcg = compute_ndcg(predictions, ground_truth, k=3)
        mrr = compute_mrr(predictions, ground_truth.keys())
        self.assertGreater(ndcg, 0)
        self.assertGreater(mrr, 0)


if __name__ == "__main__":
    unittest.main()