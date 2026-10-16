"""Dataset-native metrics, against worked examples computed by hand
(HAL-3087)."""

import pytest

from vectorless_bench.datasets.base import CORPUS
from vectorless_bench.metrics.native import NativeScorer, jaccard, squad_normalize, token_f1
from vectorless_bench.schema import AnswerType, Doc, GoldAnchor, Question, RetrievedSection


def test_squad_normalisation_and_token_f1():
    assert squad_normalize("The  Cat, sat!") == "cat sat"
    # pred "cat sat on mat" vs gold "a cat sat": common {cat, sat}; P = 2/4, R = 2/2
    assert token_f1("cat sat on mat", "a cat sat") == pytest.approx(2 * 0.5 * 1.0 / 1.5)
    assert token_f1("", "") == 1.0 and token_f1("x", "") == 0.0
    assert jaccard("the termination clause", "termination clause applies") == pytest.approx(2 / 3)


def _lbr_scorer():
    text = "0123456789" * 10  # 100 chars
    return text, NativeScorer("legalbench_rag", [Doc(doc_id="f.txt", title="f", content=text)])


def test_legalbench_character_precision_and_recall():
    text, s = _lbr_scorer()
    q = Question(qid="q", doc_id="f.txt", question="?", meta={"spans": [("f.txt", [10, 20])]})
    # one unit covering chars 15-30: overlap 5 of 15 retrieved, of 10 gold
    m = s.retrieval(q, [RetrievedSection(content=text[15:30], doc_id="f.txt")], k=5)
    assert m["native_char_precision@5"] == pytest.approx(5 / 15)
    assert m["native_char_recall@5"] == pytest.approx(5 / 10)
    assert m["native_units_unlocated"] == 0.0


def test_legalbench_unlocated_unit_counts_against_precision():
    text, s = _lbr_scorer()
    q = Question(qid="q", doc_id="f.txt", question="?", meta={"spans": [("f.txt", [10, 20])]})
    m = s.retrieval(q, [RetrievedSection(content=text[10:20]), RetrievedSection(content="not in the file at all")], k=5)
    assert m["native_char_recall@5"] == 1.0
    assert m["native_char_precision@5"] == pytest.approx(10 / (10 + len("not in the file at all")))
    assert m["native_units_unlocated"] == 1.0


def test_legalbench_overlapping_units_are_not_double_counted():
    text, s = _lbr_scorer()
    q = Question(qid="q", doc_id="f.txt", question="?", meta={"spans": [("f.txt", [0, 10])]})
    m = s.retrieval(q, [RetrievedSection(content=text[0:10]), RetrievedSection(content=text[5:10])], k=5)
    assert m["native_char_precision@5"] == 1.0 and m["native_char_recall@5"] == 1.0


def test_a_unit_with_different_spacing_is_still_located():
    doc = "Section 4.\n\n  The Licensee shall   pay all fees.\nSection 5."
    s = NativeScorer("legalbench_rag", [Doc(doc_id="d", title="d", content=doc)])
    start = doc.index("The Licensee")
    end = doc.index("fees.") + len("fees.")
    q = Question(qid="q", doc_id="d", question="?", meta={"spans": [("d", [start, end])]})
    m = s.retrieval(q, [RetrievedSection(content="The Licensee shall pay all fees.")], k=3)
    assert m["native_units_unlocated"] == 0.0 and m["native_char_recall@3"] == 1.0


def test_multihop_document_ranking():
    s = NativeScorer("multihop_rag", [])
    q = Question(qid="q", doc_id=CORPUS, question="?", meta={"doc_ids": ["a", "c"]})
    units = [RetrievedSection(content="x", doc_id=d) for d in ("b", "a", "a", "c", "d")]
    m = s.retrieval(q, units, k=5)
    # documents in rank order: b, a, c, d. Gold a at 2, c at 3.
    assert m["native_hits@5"] == 1.0
    assert m["native_mrr@5"] == pytest.approx(1 / 2)
    assert m["native_map@5"] == pytest.approx(((1 / 2) + (2 / 3)) / 2)
    m2 = s.retrieval(q, units, k=2)
    assert m2["native_hits@2"] == 0.5


def test_qasper_answer_f1_takes_the_best_annotator_and_reads_abstention():
    s = NativeScorer("qasper", [])
    q = Question(qid="q", doc_id="p", question="?", answer="BERT",
                 meta={"answer_strings": ["a BERT model", "BERT"]})
    assert s.answer(q, "BERT")["native_answer_f1"] == 1.0
    un = Question(qid="u", doc_id="p", question="?", answer_type=AnswerType.NO_ANSWER,
                  meta={"answer_strings": ["unanswerable"]})
    assert s.answer(un, "INSUFFICIENT")["native_answer_f1"] == 1.0
    assert s.answer(un, "It uses BERT")["native_answer_f1"] == 0.0


def test_qasper_evidence_f1_over_the_papers_paragraphs():
    s = NativeScorer("qasper", [])
    probes = ["we train on squad", "results improve by ten points", "related work covers parsing"]
    q = Question(qid="q", doc_id="p", question="?", meta={"paragraph_probes": probes, "evidence_paragraphs": [0, 1]})
    m = s.retrieval(q, [RetrievedSection(content="... we train on squad ..."),
                        RetrievedSection(content="related work covers parsing")], k=5)
    # predicted {0, 2}, gold {0, 1}: P = 1/2, R = 1/2
    assert m["native_evidence_precision"] == 0.5 and m["native_evidence_recall"] == 0.5
    assert m["native_evidence_f1"] == 0.5


def test_cuad_clause_match_and_no_clause_abstention():
    s = NativeScorer("cuad", [])
    q = Question(qid="q", doc_id="c", question="?", gold=[GoldAnchor(doc_id="c", answer_spans=["governed by the laws of New York"])],
                 meta={"answers": ["governed by the laws of New York"]}, answer="x")
    assert s.answer(q, "This Agreement is governed by the laws of New York.")["native_clause_match"] == 1.0
    assert s.answer(q, "INSUFFICIENT")["native_clause_match"] == 0.0
    none = Question(qid="n", doc_id="c", question="?", answer_type=AnswerType.NO_ANSWER)
    assert s.answer(none, "INSUFFICIENT")["native_clause_match"] == 1.0
    assert s.answer(none, "Section 9 covers it")["native_clause_match"] == 0.0


def test_other_datasets_have_no_native_metric_rather_than_a_zero():
    s = NativeScorer("financebench", [])
    q = Question(qid="q", doc_id="d", question="?")
    assert s.retrieval(q, [RetrievedSection(content="x")], k=5) == {}
    assert s.answer(q, "x") == {}


def test_qasper_loader_answer_strings():
    from vectorless_bench.datasets.qasper import _answer_string

    assert _answer_string({"unanswerable": True}) == "unanswerable"
    assert _answer_string({"yes_no": False}) == "no"
    assert _answer_string({"free_form_answer": " BERT "}) == "BERT"
    assert _answer_string({"free_form_answer": "", "extractive_spans": ["a", "b"]}) == "a, b"
