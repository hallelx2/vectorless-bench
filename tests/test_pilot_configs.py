"""Every pilot config loads, validates, and names only parameters its
systems really take (retrievers swallow unknown ones through **_)."""

import inspect
from pathlib import Path

import pytest

from vectorless_bench.config import load_config, validate

CONFIGS = sorted(Path(__file__).resolve().parents[1].glob("configs/pilot/*.yaml"))


def _init_params(system):
    """The keyword parameters the system's retriever class really takes."""
    import importlib
    import re

    from vectorless_bench.retrievers import registry

    src = inspect.getsource(registry._FACTORIES[system])
    module, cls_name = re.search(r"from \.(\w+) import (\w+)", src).groups()
    cls = getattr(importlib.import_module(f"vectorless_bench.retrievers.{module}"), cls_name)
    params = set(inspect.signature(cls.__init__).parameters) - {"self", "_"}
    if "embedding_model" in src:  # a factory that sets defaults before building
        params.add("embedding_model")
    return params


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_pilot_config_is_valid(path):
    cfg = load_config(str(path))
    validate(cfg)
    assert cfg.limit == 50 and cfg.judge and cfg.judge_model == "jev-latest"
    assert cfg.max_spend_usd and cfg.rig.get("probes", {}).get("engine")
    for system, params in cfg.system_params.items():
        assert system in cfg.systems, f"{path.stem}: params for {system}, which does not run"
        known = _init_params(system)
        unknown = set(params) - known
        assert not unknown, f"{path.stem}: {system} takes no {sorted(unknown)} (takes {sorted(known)})"


def test_there_is_a_pilot_config_per_dataset():
    assert {p.stem for p in CONFIGS} == {"financebench", "qasper", "cuad", "legalbench_rag",
                                         "mmlongbench_doc", "multihop_rag", "manuals"}


def test_an_unknown_config_key_is_refused(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("dataset: fixtures\nmax_spend: 40\n")
    with pytest.raises(ValueError, match="max_spend"):
        load_config(str(p))
