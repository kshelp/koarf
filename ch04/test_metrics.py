"""
test_metrics.py
================
answer_eval.py / retrieval_eval.py의 순수 함수(외부 서비스 없이 동작하는
지표 계산 함수)에 대한 단위 테스트. Ollama/Qdrant 없이 실행 가능하다.

실행 방법: (ch04 디렉터리에서)
    python3 -m unittest test_metrics.py -v
"""

import unittest

from answer_eval import cosine_similarity, exact_match, f1_score, normalize_text, rouge_l
from retrieval_eval import _dedupe_preserve_rank, dcg_at_k, ndcg_at_k, recall_at_k


class NormalizeAndExactMatchTest(unittest.TestCase):
    def test_normalize_strips_punct_and_case(self):
        self.assertEqual(normalize_text("Hello, World!"), "hello world")

    def test_exact_match_true_after_normalization(self):
        self.assertEqual(exact_match("서울시,", "서울시"), 1.0)

    def test_exact_match_false(self):
        self.assertEqual(exact_match("서울시", "부산시"), 0.0)


class F1ScoreTest(unittest.TestCase):
    def test_identical_strings_score_one(self):
        self.assertAlmostEqual(f1_score("서울시는 탄소중립을 목표로 한다", "서울시는 탄소중립을 목표로 한다"), 1.0)

    def test_disjoint_strings_score_zero(self):
        self.assertEqual(f1_score("사과 바나나", "고양이 강아지"), 0.0)

    def test_partial_overlap_between_zero_and_one(self):
        score = f1_score("서울시는 2050년까지 탄소중립을 목표로 한다", "서울시는 탄소중립을 추진한다")
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_empty_prediction_matches_empty_reference(self):
        self.assertEqual(f1_score("", ""), 1.0)


class RougeLTest(unittest.TestCase):
    def test_identical_strings_score_one(self):
        self.assertAlmostEqual(rouge_l("가나다라", "가나다라"), 1.0)

    def test_no_common_subsequence_scores_zero(self):
        self.assertEqual(rouge_l("사과 바나나", "고양이 강아지"), 0.0)


class CosineSimilarityTest(unittest.TestCase):
    def test_identical_vectors(self):
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)

    def test_orthogonal_vectors(self):
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_zero_vector_returns_zero(self):
        self.assertEqual(cosine_similarity([0.0, 0.0], [1.0, 0.0]), 0.0)


class DedupePreserveRankTest(unittest.TestCase):
    def test_keeps_first_occurrence_only(self):
        self.assertEqual(
            _dedupe_preserve_rank(["a", "b", "a", "c", "b"]),
            ["a", "b", "c"],
        )


class RecallAtKTest(unittest.TestCase):
    def test_recall_counts_unique_hits_within_k(self):
        retrieved = ["docX", "doc1", "doc2", "doc3"]
        relevant = {"doc1": 1, "doc2": 1}
        self.assertEqual(recall_at_k(retrieved, relevant, k=3), 1.0)
        self.assertEqual(recall_at_k(retrieved, relevant, k=1), 0.0)

    def test_duplicate_doc_id_does_not_double_count(self):
        # ch03가 doc_id를 "파일명#페이지" 단위로 성기게 잡을 때 같은 문서가 두 번
        # 뽑혀도 recall이 1.0을 넘지 않아야 한다(dedup 회귀 테스트).
        retrieved = ["doc1", "doc1", "doc2"]
        relevant = {"doc1": 1, "doc2": 1}
        self.assertLessEqual(recall_at_k(retrieved, relevant, k=3), 1.0)

    def test_empty_relevant_docs_returns_zero(self):
        self.assertEqual(recall_at_k(["doc1"], {}, k=3), 0.0)


class NdcgAtKTest(unittest.TestCase):
    def test_perfect_ranking_scores_one(self):
        retrieved = ["doc1", "doc2"]
        relevant = {"doc1": 2, "doc2": 1}
        self.assertAlmostEqual(ndcg_at_k(retrieved, relevant, k=2), 1.0)

    def test_reversed_ranking_scores_less_than_one(self):
        retrieved = ["doc2", "doc1"]
        relevant = {"doc1": 2, "doc2": 1}
        self.assertLess(ndcg_at_k(retrieved, relevant, k=2), 1.0)

    def test_no_relevant_docs_returns_zero(self):
        self.assertEqual(ndcg_at_k(["doc1"], {}, k=1), 0.0)

    def test_dcg_ignores_irrelevant_docs(self):
        self.assertEqual(dcg_at_k(["docX"], {"doc1": 1}, k=1), 0.0)


if __name__ == "__main__":
    unittest.main()
