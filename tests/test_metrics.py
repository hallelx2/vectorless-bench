import math

from vectorless_bench.metrics import aggregate, retrieval
from vectorless_bench.metrics.citation import citation_metrics, primary_quality
from vectorless_bench.schema import GoldAnchor, RetrievedSection


def test_precision_recall_f1():
    labels = [True, False, True, False]
    assert retrieval.precision_at_k(labels, 4) == 0.5
    assert retrieval.recall_at_k(labels, 4, num_gold=2) == 1.0
    assert math.isclose(retrieval.f1_at_k(labels, 4, 2), 2 * 0.5 * 1.0 / 1.5)


def test_mrr_and_hit():
    assert retrieval.mrr([False, True, True]) == 0.5
    assert retrieval.mrr([False, False]) == 0.0
    assert retrieval.hit_at_k([False, True], 2) == 1.0
    assert retrieval.hit_at_k([False, False], 2) == 0.0


def test_ndcg_perfect_is_one():
    # all relevant up front -> nDCG 1.0
    assert math.isclose(retrieval.ndcg_at_k([True, True], 2, num_gold=2), 1.0)
    # a relevant item lower down scores less than at the top
    assert retrieval.ndcg_at_k([False, True], 2, 1) < 1.0


def test_percentiles_monotonic():
    p = aggregate.percentiles([10, 20, 30, 40, 50])
    assert p["p50"] <= p["p95"] <= p["p99"] <= p["max"]
    assert p["mean"] == 30


def test_determinism_exact_and_jaccard():
    same = aggregate.determinism([["a", "b"], ["a", "b"], ["a", "b"]])
    assert same["exact_match"] == 1.0 and same["mean_jaccard"] == 1.0
    diff = aggregate.determinism([["a", "b"], ["a", "c"]])
    assert diff["exact_match"] == 0.0
    assert 0.0 < diff["mean_jaccard"] < 1.0


def test_cost_summary_quality_per_dollar():
    s = aggregate.cost_summary(
        costs=[0.001, 0.001], qualities=[1.0, 0.0],
        total_tokens=[100, 100], llm_calls=[1, 1],
    )
    assert s["mean_cost_usd"] == 0.001
    assert s["cost_per_correct_usd"] == 0.002  # 1 correct of 2
    assert s["quality_per_1k_usd"] > 0


def test_near_miss_metric():
    gold = [GoldAnchor(doc_id="d", title_path=["P", "FY2024"], answer_spans=["13.8%"])]
    secs = [RetrievedSection(content="12.1%", title_path=["P", "FY2023"])]
    m = citation_metrics(secs, gold)
    assert m["near_miss_rate"] == 1.0
    assert m["span_in_top1"] == 0.0


def test_path_correct_is_structural_only():
    # A chunk system matches the span but has NO path -> path_correct must be 0,
    # even though it found the answer text. This is the structural differentiator.
    gold = [GoldAnchor(doc_id="d", title_path=["Item 8"], answer_spans=["13.8%"])]
    chunk = [RetrievedSection(content="CET1 was 13.8%")]  # no title_path
    structural = [RetrievedSection(content="CET1 was 13.8%", title_path=["Part II", "Item 8"])]
    assert citation_metrics(chunk, gold)["span_in_top1"] == 1.0
    assert citation_metrics(chunk, gold)["path_correct_top1"] == 0.0
    assert citation_metrics(structural, gold)["path_correct_top1"] == 1.0


def test_primary_quality_picks_abstention_for_no_answer():
    assert primary_quality({"abstained": 1.0}, k=5) == 1.0
    assert primary_quality({"f1@5": 0.7}, k=5) == 0.7


def test_primary_quality_prefers_judged_answer_correctness():
    # When the LLM-judge axis ran, judged answer-correctness is the headline and
    # supersedes BOTH span-F1 and the abstention flag. An answer-first system can
    # have near-zero span-overlap F1 yet a correct answer; ranking on F1 would
    # wrongly bury it. This is the bench-correctness fix.
    assert primary_quality(
        {"f1@5": 0.1, "abstained": 0.0, "answer_correct": 1.0}, k=5
    ) == 1.0
    assert primary_quality({"f1@5": 0.9, "answer_correct": 0.0}, k=5) == 0.0
    # No judge -> fall back to F1@k (unjudged configs unchanged).
    assert primary_quality({"f1@5": 0.42}, k=5) == 0.42
