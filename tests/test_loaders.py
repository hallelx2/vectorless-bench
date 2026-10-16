"""Each loader against a miniature of its real file shape."""

import json
import zipfile

from vectorless_bench.datasets import CORPUS, build_dataset
from vectorless_bench.datasets._files import PAGE_BREAK, probe
from vectorless_bench.schema import AnswerType


def test_multihop_rag(tmp_path):
    d = tmp_path / "mh"
    d.mkdir()
    (d / "corpus.json").write_text(json.dumps([
        {"title": "Apple earnings", "body": "Apple said revenue rose 8% in Q3.", "source": "X", "published_at": "2023-10-01"},
        {"title": "Fed holds rates", "body": "The Fed held rates at 5.25%.", "source": "Y", "published_at": "2023-10-02"},
    ]))
    (d / "MultiHopRAG.json").write_text(json.dumps([
        {"query": "q1", "answer": "Yes", "question_type": "comparison_query",
         "evidence_list": [{"title": "Apple earnings", "fact": "revenue rose 8% in Q3"},
                           {"title": "Fed holds rates", "fact": "held rates at 5.25%"}]},
        {"query": "q2", "answer": "Insufficient information.", "question_type": "null_query", "evidence_list": []},
    ]))
    ds = build_dataset("multihop_rag", data_dir=str(d))
    qs = ds.questions()
    assert qs[0].doc_id == CORPUS and len(qs[0].meta["doc_ids"]) == 2 and qs[0].answer_type == AnswerType.MULTIHOP
    assert qs[1].answer_type == AnswerType.NO_ANSWER and not qs[1].gold
    c = ds.check()
    assert c["corpus_level_questions"] == 2 and c["span_found_pct"] == 100.0


def test_legalbench_rag_spans_and_mini(tmp_path):
    d = tmp_path / "lb"
    (d / "corpus" / "cuad").mkdir(parents=True)
    text = "Section 1. Term. This Agreement expires on December 31, 2025."
    (d / "corpus" / "cuad" / "a.txt").write_text(text)
    (d / "benchmarks").mkdir()
    tests = [{"query": f"Consider the agreement; q{i}", "snippets": [{"file_path": "cuad/a.txt", "span": [11, 62], "answer": ""}]} for i in range(300)]
    (d / "benchmarks" / "cuad.json").write_text(json.dumps({"tests": tests}))
    ds = build_dataset("legalbench_rag", data_dir=str(d), subsets=["cuad"])
    qs = ds.questions()
    assert len(qs) == 300 and qs[0].gold[0].answer_spans[0].startswith("Term.")
    assert len(build_dataset("legalbench_rag", data_dir=str(d), subsets=["cuad"], mini=True).questions()) == 194
    assert ds.check()["span_found_pct"] == 100.0


def test_cuad_no_clause_is_no_answer(tmp_path):
    d = tmp_path / "cuad"
    (d / "pdfs").mkdir(parents=True)
    (d / "test.json").write_text(json.dumps({"data": [{"title": "K1", "paragraphs": [{"context": "", "qas": [
        {"id": "K1__Governing Law", "question": "gov law?", "is_impossible": False, "answers": [{"text": "laws of New York", "answer_start": 0}]},
        {"id": "K1__Non-Compete", "question": "non compete?", "is_impossible": True, "answers": []},
    ]}]}]}))
    (d / "pdfs" / "K1.txt").write_text("This Agreement is governed by the laws of New York." + PAGE_BREAK + "page two")
    ds = build_dataset("cuad", data_dir=str(d))
    qs = ds.questions()
    assert qs[0].meta["clause_type"] == "Governing Law" and qs[1].answer_type == AnswerType.NO_ANSWER
    assert ds.check()["span_found_pct"] == 100.0


def test_mmlongbench_pages(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    d = tmp_path / "mm"
    (d / "documents").mkdir(parents=True)
    pq.write_table(pa.table({"doc_id": ["r.pdf", "r.pdf"], "doc_type": ["Report", "Report"],
                             "question": ["share?", "unknown?"], "answer": ["42%", "Not answerable"],
                             "evidence_pages": ["[2]", "[]"], "evidence_sources": ["['Chart']", "[]"],
                             "answer_format": ["Str", "None"]}), d / "questions.parquet")
    body = " ".join(f"token{i}" for i in range(40)) + " the share was 42% in 2023"
    (d / "documents" / "r.txt").write_text("cover" + PAGE_BREAK + body)
    ds = build_dataset("mmlongbench_doc", data_dir=str(d))
    qs = ds.questions()
    assert qs[0].gold[0].page == 2 and qs[0].domain == "chart" and qs[1].answer_type == AnswerType.NO_ANSWER
    c = ds.check()
    assert c["page_text_anchors"] == 1 and c["page_text_matchable_pct"] == 100.0
    assert qs[0].gold[0].page_text == body and not qs[0].gold[0].answer_spans  # the answer is not evidence


def test_manuals_reads_malformed_json_and_builds_headings(tmp_path):
    d = tmp_path / "man"
    d.mkdir()
    bad = '{"All_Questions": [{"Section Hierarchy": "Quick Guides>Using Bixby>Running Bixby", "Link": "x?page=14", "Question": "Where is the "Bixby" guide?", "Answer": "Press and hold the voice button on your Samsung Smart Remote, say a command, and then release the button to run Bixby."}]}'
    with zipfile.ZipFile(d / "smart_remote_QA.zip", "w") as z:
        z.writestr("Normal Q&A/n.json", bad)
        z.writestr("Paraphrased Q&A/p.json", bad)
    (d / "corpus.json").write_text(json.dumps({"s0": {"title": "Quick Guides", "text": ["Intro."]},
        "s1": {"title": "Running Bixby", "text": ["Press and hold the voice button on your Samsung Smart Remote, say a command, and then release the button to run Bixby."]}}))
    ds = build_dataset("manuals", data_dir=str(d))
    qs = ds.questions()
    assert len(qs) == 1, "paraphrased duplicates are left out"
    assert qs[0].gold[0].title_path == ["Quick Guides", "Using Bixby", "Running Bixby"]
    doc = ds.corpus()[0]
    assert "#### Running Bixby" in doc.content and "## Quick Guides" in doc.content
    assert ds.check()["span_found_pct"] == 100.0


def test_qasper_evidence_window_and_page(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    d = tmp_path / "qa"
    (d / "pdfs").mkdir(parents=True)
    para = ("We collected a corpus of human human dialogues on the persuasion task and annotated each turn "
            "with dialogue acts so that the strategies could be compared across settings")
    row = {"id": "1909.00001", "title": "T", "abstract": "", "full_text": {"section_name": [], "paragraphs": []},
           "qas": {"question": ["How big is the corpus?", "Is it public?"], "question_id": ["q1", "q2"],
                   "answers": [{"answer": [{"unanswerable": False, "extractive_spans": ["1,017 dialogues"], "yes_no": None,
                                            "free_form_answer": "", "evidence": [para], "highlighted_evidence": []}]},
                               {"answer": [{"unanswerable": True, "extractive_spans": [], "yes_no": None,
                                            "free_form_answer": "", "evidence": [], "highlighted_evidence": []}]}]}}
    pq.write_table(pa.Table.from_pylist([row]), d / "test.parquet")
    (d / "pdfs" / "1909.00001.txt").write_text("Title page" + PAGE_BREAK + "Section 3. " + para.replace("human human", "human-\nhuman"))
    ds = build_dataset("qasper", data_dir=str(d))
    qs = ds.questions()
    assert qs[0].answer == "1,017 dialogues" and qs[0].gold[0].page == 2
    assert qs[1].answer_type == AnswerType.NO_ANSWER
    assert len(probe(para).split()) == 12


def test_gold_reachable_marks_questions_no_system_can_win(tmp_path):
    from vectorless_bench.datasets.base import gold_reachable
    from vectorless_bench.schema import Doc, GoldAnchor, Question

    docs = {"d": Doc(doc_id="d", title="d", content="the revenue grew to 1,577 million in fiscal 2023")}
    found = Question(qid="a", doc_id="d", question="?", gold=[GoldAnchor(doc_id="d", answer_spans=["revenue grew to 1,577 million"])], answer="x")
    lost = Question(qid="b", doc_id="d", question="?", gold=[GoldAnchor(doc_id="d", answer_spans=["a chart of churn by region"])], answer="x")
    figure = Question(qid="c", doc_id="d", question="?", gold=[GoldAnchor(doc_id="d", page=4, page_text="Figure 3")], answer="x")
    assert gold_reachable(found, docs) and not gold_reachable(lost, docs) and not gold_reachable(figure, docs)


def test_a_corpus_question_keeps_the_whole_corpus(monkeypatch):
    """A sampled MultiHop run must still search all the articles: trimming
    the corpus to the sampled questions' gold documents would hand every
    system a library made of answers."""
    from vectorless_bench import runner
    from vectorless_bench.config import RunConfig
    from vectorless_bench.datasets.base import CORPUS
    from vectorless_bench.schema import Doc, GoldAnchor, Question

    docs = [Doc(doc_id=f"n{i}", title=f"n{i}", content="x") for i in range(10)]

    class _DS:
        def corpus(self):
            return docs

        def questions_for_available_corpus(self):
            return [Question(qid=f"q{i}", doc_id=CORPUS, question="?", meta={"doc_ids": [f"n{i}", f"n{i + 1}"]},
                             gold=[GoldAnchor(doc_id=f"n{i}", answer_spans=["x"])], answer="x") for i in range(4)]

        def subset(self, qs, limit, seed=0):
            return qs[:limit]

    monkeypatch.setattr(runner, "build_dataset", lambda *a, **k: _DS())
    _, corpus, questions = runner._load_dataset(RunConfig(dataset="multihop_rag", limit=1))
    assert len(questions) == 1 and len(corpus) == 10
