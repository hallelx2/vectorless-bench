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


def test_a_nested_set_up_is_worked_out_inside_out():
    # FinanceBench 04412 (Lockheed Martin), as the model wrote it
    text, calcs = resolve("Ratio = <<65,398 / <<(50,710 + 47,528) / 2>>>>")
    assert text == "Ratio = 1.3314 (= 65,398 / 49119)"
    assert [c.expression for c in calcs] == ["(50,710 + 47,528) / 2", "65,398 / 49119"]
    assert calcs[1].value == pytest.approx(65398 / 49119)


def test_two_levels_and_siblings():
    text, calcs = resolve("x <<<<1 + 1>> * <<2 + 3>> + <<<<4>> / 2>>>> y")
    assert text == "x 12 (= 2 * 5 + 2) y" and len(calcs) == 5


def test_a_failed_inner_set_up_fails_the_outer_one():
    text, calcs = resolve("<<10 / <<abs(1)>>>>")
    assert text == "10 / (abs(1)) (could not be computed)"
    assert all(c.value is None for c in calcs)


@pytest.mark.parametrize("raw,want", [
    ("margin <<11,512 / 86,392 * 100", "margin 11,512 / 86,392 * 100 (could not be computed)"),
    ("margin 11,512 / 86,392 * 100>> %", "margin 11,512 / 86,392 * 100 %"),
])
def test_an_unbalanced_marker_is_never_a_number(raw, want):
    text, calcs = resolve(raw)
    assert text == want and calcs and calcs[-1].value is None


def test_an_inner_result_keeps_its_precision():
    text, _ = resolve("<<<<1 / 3>> * 3>>")
    assert text == "1 (= 0.3333333333333333 * 3)"
    # small and large inner results keep every digit (no rounding to 12 places)
    assert resolve("<<1 / <<0.0000000000001 * 1>>>>")[1][-1].value == pytest.approx(1e13)
    assert resolve("<<<<1 / 3>> * 1000000000000>>")[1][-1].value == pytest.approx(1e12 / 3)
