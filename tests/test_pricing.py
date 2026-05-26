from vectorless_bench import pricing


def test_compute_matches_engine_formula():
    # claude-sonnet-4-5 is 3.00 / 15.00 per Mtok (mirrors llmgate pricing.go)
    cost = pricing.compute("claude-sonnet-4-5", 1_000_000, 1_000_000)
    assert cost == 3.00 + 15.00


def test_unknown_model_is_free_unless_strict():
    assert pricing.compute("does-not-exist", 1000, 1000) == 0.0


def test_embedding_cost():
    # text-embedding-3-small = $0.02 / Mtok
    assert pricing.compute_embedding("text-embedding-3-small", 1_000_000) == 0.02


def test_count_tokens_nonzero_and_fallback():
    assert pricing.count_tokens("", "gpt-4o") == 0
    assert pricing.count_tokens("hello world foo bar", "gpt-4o") > 0


def test_fingerprint_is_stable():
    assert pricing.pricing_fingerprint() == pricing.pricing_fingerprint()
    assert len(pricing.pricing_fingerprint()) == 12
