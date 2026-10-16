import pytest

from vectorless_bench import _llm
from vectorless_bench.config import RunConfig, validate


def test_defaults_are_vertex():
    validate(RunConfig())


def test_openai_model_is_refused():
    with pytest.raises(ValueError, match="gpt-4o"):
        validate(RunConfig(model="gpt-4o"))


def test_system_param_models_are_checked():
    cfg = RunConfig(system_params={"pageindex": {"model": "claude-haiku-4-5"}})
    with pytest.raises(ValueError, match="pageindex"):
        validate(cfg)


def test_local_bge_and_jev_are_allowed():
    validate(RunConfig(embedding_model="BAAI/bge-m3", judge=True, judge_model="jev-latest"))


def test_require_vertex_model():
    _llm.require_vertex_model("gemini-2.5-flash-lite")
    with pytest.raises(ValueError):
        _llm.require_vertex_model("gpt-4o-mini")


class _UM:
    prompt_token_count = 1000
    cached_content_token_count = 400
    candidates_token_count = 20
    thoughts_token_count = 56
    tool_use_prompt_token_count = 0
    total_token_count = 1076


def test_usage_from_vertex_metadata():
    u = _llm.usage_from("gemini-2.5-flash-lite", _UM())
    assert u.output_tokens == 76  # thinking bills as output
    assert u.input_tokens == 1000
    want = (600 * 0.10 + 76 * 0.40 + 400 * 0.01) / 1e6
    assert abs(u.cost_usd - want) < 1e-12


def test_thinking_budget_env(monkeypatch):
    monkeypatch.delenv("VLBENCH_THINKING_BUDGET", raising=False)
    assert _llm.thinking_budget() == 0
    monkeypatch.setenv("VLBENCH_THINKING_BUDGET", "")
    assert _llm.thinking_budget() is None
    monkeypatch.setenv("VLBENCH_THINKING_BUDGET", "512")
    assert _llm.thinking_budget() == 512
