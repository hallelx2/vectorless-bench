"""Arithmetic for the shared answer step, done in code (HAL-3203).

The answer model sets a calculation up from figures in the excerpts and
writes it as ``<<expression>>``; this module evaluates it and puts the result
in its place. Measured on the pilot's FinanceBench declines: given the right
inputs, gemini-2.5-flash-lite wrote ``365 * 1,380.5 / 7,890 = 67.75`` (it is
63.86), so a grade on its arithmetic grades the model, not what the system
retrieved.

Only numbers, + - * /, parentheses and unary signs are accepted. Thousands
separators, currency signs and a trailing % on a number are read the way a
filing writes them ("$1,174", "12.5%" is 0.125).
"""

from __future__ import annotations

import ast
import math
import operator
import re
from dataclasses import dataclass
from typing import List, Tuple

_EXPR = re.compile(r"<<(.+?)>>", re.S)
# A currency sign may be followed by a space; a bare number may not be
# preceded by one (so "5 3" stays two numbers and fails). Thousands
# separators only in real groups: "1,174" is 1174, "1,2" and "1, 2" are not
# numbers and fail rather than read as 12, and "1,2345" is not 12345.
_NUMBER = re.compile(r"(?:[$€£]\s*)?(?:\d{1,3}(?:,\d{3})+(?![\d])|\d+)(?:\.\d+)?%?|(?:[$€£]\s*)?\.\d+%?")
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


class CalcError(ValueError):
    pass


@dataclass
class Calculation:
    expression: str
    value: float | None
    error: str = ""


def _normalise(expr: str) -> str:
    def num(m: re.Match) -> str:
        s = re.sub(r"[$€£,\s]", "", m.group(0))
        if re.match(r"0\d", s):  # "007" is not a Python literal
            s = s.lstrip("0") or "0"
        if s.endswith("%"):
            return f"({s[:-1]}/100)"
        return s

    e = expr.replace("×", "*").replace("÷", "/").replace("−", "-").replace("–", "-")
    return _NUMBER.sub(num, e)


def evaluate(expr: str) -> float:
    """The value of one arithmetic expression; CalcError when it is not one."""
    try:
        tree = ast.parse(_normalise(expr), mode="eval")
    except (SyntaxError, ValueError, RecursionError, MemoryError) as e:
        # model output is untrusted: a null byte, deep nesting or a huge
        # literal must say "could not be computed", not abort the grade
        raise CalcError(f"not an arithmetic expression: {expr[:80]!r}") from e

    def ev(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            try:
                return float(node.value)
            except OverflowError as e:
                raise CalcError("number too large") from e
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Div) and right == 0:
                raise CalcError("division by zero")
            return _OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            v = ev(node.operand)
            return v if isinstance(node.op, ast.UAdd) else -v
        raise CalcError(f"not allowed in a calculation: {ast.dump(node)[:60]}")

    try:
        v = ev(tree)
    except RecursionError as e:
        raise CalcError("nested too deeply") from e
    if math.isnan(v) or math.isinf(v):
        raise CalcError("not a finite number")
    return v


def fmt(v: float) -> str:
    """A result as a reader writes it: no float noise, up to four decimals."""
    if v == int(v) and abs(v) < 1e15:
        return f"{int(v):,}"
    return f"{v:,.4f}".rstrip("0").rstrip(".")


def _plain(v: float) -> str:
    """An inner result written back into its outer expression at full
    precision, in a form the number grammar reads (no exponent)."""
    s = f"{v:.12f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-0") else "0"


def resolve(text: str) -> Tuple[str, List[Calculation]]:
    """Replace every <<expression>> in text with its value. A calculation
    that does not evaluate is left visible as the expression with a note, so
    the grade sees that the answer has no number rather than a wrong one.

    A set-up may nest another (<<65,398 / <<(50,710 + 47,528) / 2>>>>): the
    innermost is worked out first and its value feeds the one around it
    (HAL-3245). Every level is recorded. An inner one that fails makes its
    outer one fail. A << with no >>, or a >> with no <<, is not computable,
    never a partial number."""
    calcs: List[Calculation] = []
    buf: List[str] = []
    opens: List[int] = []  # where each still-open << starts in buf
    i, n = 0, len(text)
    while i < n:
        two = text[i:i + 2]
        if two == "<<":
            opens.append(len(buf))
            i += 2
            continue
        if two == ">>":
            i += 2
            if not opens:
                calcs.append(Calculation(">>", None, "a >> with no <<"))
                continue
            start = opens.pop()
            expr = "".join(buf[start:]).strip()
            del buf[start:]
            try:
                v = evaluate(expr)
            except CalcError as e:
                calcs.append(Calculation(expr, None, str(e)))
                # inside another set-up the text stays, so the outer one fails too
                buf.append(f"({expr})" if opens else f"{expr} (could not be computed)")
                continue
            calcs.append(Calculation(expr, v))
            buf.append(_plain(v) if opens else f"{fmt(v)} (= {expr})")
            continue
        buf.append(text[i])
        i += 1
    while opens:  # a << never closed: everything after it is one failed set-up
        start = opens.pop()
        expr = "".join(buf[start:]).strip()
        del buf[start:]
        calcs.append(Calculation(expr, None, "a << with no >>"))
        buf.append(f"{expr} (could not be computed)")
    return "".join(buf), calcs
