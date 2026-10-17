import pytest

from vectorless_bench.calc import CalcError, evaluate, fmt, resolve


@pytest.mark.parametrize("expr,want", [
    ("2112 / 14082 * 100", 14.99787),
    ("365 * ((34,616 + 25,309) / 2) / (111,934 + (16,047 - 11,461))", 93.8578),
    ("$1,174 + $1,587", 2761.0),
    ("(14.998 + 4.503 + 11.354) / 3", 10.285),
    ("12.5% * 200", 25.0),
    ("-3 + 5", 2.0),
    ("6 × 7 ÷ 2 − 1", 20.0),
    ("007 + 1", 8.0),
])
def test_evaluate(expr, want):
    assert evaluate(expr) == pytest.approx(want, rel=1e-4)


@pytest.mark.parametrize("expr", ["__import__('os')", "2 ** 10", "1 / 0", "abs(-1)", "x + 1", "", "1 +",
                                  "5 3", "1, 2", "1,2", "1,2345", "1\x00 + 2", "(" * 5000 + "1" + ")" * 5000,
                                  "9" * 400 + " * 1"])
def test_only_arithmetic_is_evaluated(expr):
    with pytest.raises(CalcError):
        evaluate(expr)


def test_resolve_replaces_each_calculation():
    text, calcs = resolve("Margin <<11,512 / 86,392 * 100>>%, growth <<(86,392 - 79,474) / 79,474 * 100>>%, bad <<abs(1)>>")
    assert text.startswith("Margin 13.3253 (= 11,512 / 86,392 * 100)%, growth 8.7047 (=")
    assert "bad abs(1) (could not be computed)" in text
    assert [c.value is None for c in calcs] == [False, False, True]


def test_fmt():
    assert fmt(1234.0) == "1,234" and fmt(0.1 + 0.2) == "0.3" and fmt(-3.70004) == "-3.7"
