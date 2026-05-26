from vectorless_bench import anchors
from vectorless_bench.schema import GoldAnchor, RetrievedSection


def test_span_present_formatting_robust():
    content = "The CET1 ratio was $1,234.50 at year end."
    assert anchors.span_present("$1,234.50", content)
    assert anchors.span_present("1234.5", content)  # number-normalised
    assert anchors.span_present("CET1 ratio", content)  # whitespace/case
    assert not anchors.span_present("9999", content)


def test_path_matches_suffix_and_ordinal_noise():
    gold = ["Item 8", "Balance Sheet"]
    cand = ["Part II", "Item 8.", "Balance Sheet"]
    assert anchors.path_matches(gold, cand)
    # wrong leaf under same parent must not match
    assert not anchors.path_matches(gold, ["Part II", "Item 8", "Income Statement"])


def test_match_anchor_any_dimension():
    sec = RetrievedSection(content="CET1 was 13.8%", title_path=["Part II", "Item 8"])
    by_span = GoldAnchor(doc_id="d", answer_spans=["13.8%"])
    by_path = GoldAnchor(doc_id="d", title_path=["Item 8"])
    miss = GoldAnchor(doc_id="d", answer_spans=["99%"], title_path=["Other"])
    assert anchors.match_anchor(sec, by_span)
    assert anchors.match_anchor(sec, by_path)
    assert not anchors.match_anchor(sec, miss)


def test_sibling_near_miss():
    gold = GoldAnchor(
        doc_id="d", title_path=["Part II", "FY2024"], answer_spans=["13.8%"]
    )
    sibling = RetrievedSection(content="12.1%", title_path=["Part II", "FY2023"])
    correct = RetrievedSection(content="13.8%", title_path=["Part II", "FY2024"])
    assert anchors.is_sibling_near_miss(sibling, [gold])
    assert not anchors.is_sibling_near_miss(correct, [gold])


def test_relevance_labels_order_preserved():
    golds = [GoldAnchor(doc_id="d", answer_spans=["target"])]
    secs = [
        RetrievedSection(content="nope"),
        RetrievedSection(content="this has the target inside"),
    ]
    assert anchors.relevance_labels(secs, golds) == [False, True]
