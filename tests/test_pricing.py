from vectorless_bench import pricing


def test_compute_known_model():
    # gemini-2.5-flash-lite: $0.10 in, $0.40 out per 1M (Vertex list, 2026-10-16)
    cost = pricing.compute("gemini-2.5-flash-lite", 1_000_000, 1_000_000)
    assert abs(cost - 0.50) < 1e-9


def test_cached_tokens_bill_at_the_cached_rate():
    # 1M uncached in + 1M cached: $0.10 + $0.01
    cost = pricing.compute("gemini-2.5-flash-lite", 1_000_000, 0, cached_tokens=1_000_000)
    assert abs(cost - 0.11) < 1e-9


def test_lite_is_not_priced_as_flash():
    assert pricing.lookup("gemini-2.5-flash-lite")[0] == 0.10
    assert pricing.lookup("gemini-2.5-flash-lite-001")[0] == 0.10
    assert pricing.lookup("gemini-2.5-flash")[0] == 0.30


def test_unknown_model_costs_zero_unless_strict():
    assert pricing.compute("no-such-model", 1000, 1000) == 0.0
    pricing.STRICT = True
    try:
        try:
            pricing.compute("no-such-model", 1, 1)
            raise AssertionError("STRICT should raise")
        except KeyError:
            pass
    finally:
        pricing.STRICT = False


def test_embedding_cost():
    assert abs(pricing.compute_embedding("gemini-embedding-2", 1_000_000) - 0.20) < 1e-9
    assert pricing.compute_embedding("BAAI/bge-m3", 1_000_000) == 0.0


def test_count_tokens():
    assert pricing.count_tokens("") == 0
    assert pricing.count_tokens("hello world foo bar") > 0
