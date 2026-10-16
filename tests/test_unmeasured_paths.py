from vectorless_bench.metrics.citation import citation_metrics
from vectorless_bench.report import _path_cells
from vectorless_bench.schema import AnswerType, GoldAnchor, Question, RetrievedSection


def _q(path):
    return Question(qid="q", doc_id="d", question="?", gold=[GoldAnchor(doc_id="d", title_path=path, answer_spans=["x"])],
                    answer_type=AnswerType.LOOKUP)


def test_chunk_system_has_no_path_metrics():
    m = citation_metrics([RetrievedSection(content="x")], _q(["A", "B"]).gold)
    assert "path_correct_top1" not in m and "near_miss_rate" not in m
    assert m["gold_has_path"] == 1.0


def test_no_gold_path_means_unscored():
    m = citation_metrics([RetrievedSection(content="x", title_path=["A"])], _q([]).gold)
    assert "path_correct_top1" not in m and m["gold_has_path"] == 0.0


def test_structural_system_is_scored():
    m = citation_metrics([RetrievedSection(content="x", title_path=["A", "B"])], _q(["A", "B"]).gold)
    assert m["path_correct_top1"] == 1.0
    assert m["near_miss_rate"] == 0.0


def test_cells_explain_why():
    assert _path_cells({"gold_has_path": None})[0].startswith("unknown")
    assert _path_cells({"gold_has_path": 0.0})[0] == "n/a (no gold paths)"
    assert _path_cells({"gold_has_path": 1.0, "path_correct_top1": None})[0] == "n/a (no paths returned)"
    assert _path_cells({"gold_has_path": 1.0, "path_correct_top1": 0.5, "near_miss_rate": 0.1})[0] != "n/a"
