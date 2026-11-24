"""Unit tests for F2 lazy PluginRegistry anti-shadowing and F3 load_plugins."""

from __future__ import annotations

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import pytest

from eval_studio.core.registry import (
    BUILTIN_DISTRIBUTION_NAME,
    PluginLoadError,
    PluginRegistry,
    get_registry,
    load_plugins,
)
from eval_studio.scoring.metrics import acc


def test_registry_lazy_discovery_does_not_shadow_builtins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDist:
        name = "third-party-plugin"

    class FakeEntryPoint:
        def __init__(self, name: str, value: str, dist: Any = None) -> None:
            self.name = name
            self.value = value
            self.dist = dist
            self.loaded = False

        def load(self) -> Any:
            self.loaded = True
            return lambda: self.value

    builtin_ep = FakeEntryPoint(
        "mock",
        "eval_studio.gateway.mock:MockGateway",
        dist=type("BuiltinDist", (), {"name": BUILTIN_DISTRIBUTION_NAME})(),
    )
    external_collision = FakeEntryPoint("mock", "evil_pkg:FakeMock", dist=FakeDist())
    external_valid = FakeEntryPoint("custom_ext", "custom_pkg:CustomModel", dist=FakeDist())

    monkeypatch.setattr(
        importlib_metadata,
        "entry_points",
        lambda *, group: [external_collision, builtin_ep, external_valid],
    )

    reg: PluginRegistry[Any] = PluginRegistry("eval_studio.models")
    reg.discover()

    # Discovery must remain lazy until .get() is called
    assert builtin_ep.loaded is False
    assert external_collision.loaded is False
    assert external_valid.loaded is False
    assert reg.list_names() == ["custom_ext", "mock"]

    # Registering a non-builtin with a built-in name must not shadow it
    reg.register("mock", "shadow_attempt", builtin=False)
    resolved_mock = reg.get("mock")
    assert builtin_ep.loaded is True
    assert external_collision.loaded is False
    assert resolved_mock() == "eval_studio.gateway.mock:MockGateway"

    assert reg.get("custom_ext")() == "custom_pkg:CustomModel"


def test_registry_thread_safe_concurrent_lookup() -> None:
    reg = get_registry("eval_studio.filters")
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: reg.get("multi_choice_regex"), range(32)))
    assert len({id(item) for item in results}) == 1


def test_load_plugins_requires_trust_remote_code(tmp_path: Path) -> None:
    plugin_file = tmp_path / "my_plugin.py"
    plugin_file.write_text("EVAL_STUDIO_PLUGINS = {}\n", encoding="utf-8")

    with pytest.raises(PluginLoadError, match="trust-remote-code"):
        load_plugins([str(plugin_file)], trust_remote_code=False)


def test_load_plugins_from_py_file_without_shadowing_builtin(tmp_path: Path) -> None:
    plugin_file = tmp_path / "custom_m1_plugin.py"
    plugin_file.write_text(
        "def custom_metric(x: float) -> float:\n"
        "    return x * 2.0\n\n"
        "def shadow_acc(*args, **kwargs) -> float:\n"
        "    return -999.0\n\n"
        "EVAL_STUDIO_PLUGINS = {\n"
        "    'eval_studio.metrics': {\n"
        "        'custom_m1_double': custom_metric,\n"
        "        'acc': shadow_acc,\n"
        "    }\n"
        "}\n",
        encoding="utf-8",
    )

    loaded = load_plugins([str(plugin_file)], trust_remote_code=True)
    assert loaded == [str(plugin_file)]
    metrics_reg = get_registry("eval_studio.metrics")
    assert metrics_reg.get("custom_m1_double")(3.5) == 7.0
    # Built-in 'acc' must never be shadowed by custom plugins
    assert metrics_reg.get("acc") is acc


def test_spoofed_eval_studio_prefix_external_entry_point_cannot_shadow_builtin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDist:
        name = "evil-third-party"

    class FakeEntryPoint:
        def __init__(self, name: str, value: str, dist: Any = None) -> None:
            self.name = name
            self.value = value
            self.dist = dist
            self.loaded = False

        def load(self) -> Any:
            self.loaded = True
            return lambda: "evil"

    spoof_ep = FakeEntryPoint("mock", "eval_studio.evil:EvilMock", dist=FakeDist())
    monkeypatch.setattr(importlib_metadata, "entry_points", lambda *, group: [spoof_ep])

    reg: PluginRegistry[Any] = PluginRegistry("eval_studio.models")
    reg.register_lazy_builtin("mock", "eval_studio.gateway.mock:MockGateway")
    reg.discover()

    resolved = reg.get("mock")
    assert spoof_ep.loaded is False
    assert resolved.__name__ == "MockGateway"


def test_core_registry_list_names_is_lazy_with_zero_gateway_or_heavy_imports() -> None:
    probe = (
        "import sys; "
        "import eval_studio.core; "
        "from eval_studio.core.registry import ENTRY_POINT_GROUPS, get_registry; "
        "names = {g: get_registry(g).list_names() for g in ENTRY_POINT_GROUPS}; "
        "assert 'mock' in names['eval_studio.models']; "
        "forbidden = ('eval_studio.gateway', 'eval_studio.scoring', 'eval_studio.stats', "
        "'torch', 'transformers', 'vllm', 'docker', 'pypdf'); "
        "loaded = [m for m in sys.modules if m.startswith(forbidden)]; "
        "assert not loaded, f'Eagerly loaded: {loaded}'"
    )
    res = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stderr or res.stdout
