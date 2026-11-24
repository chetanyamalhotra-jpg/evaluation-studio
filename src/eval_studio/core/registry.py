"""Thread-safe lazy plugin registry backed by importlib.metadata entry points."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from collections.abc import Mapping, Sequence
from importlib import metadata as importlib_metadata
from pathlib import Path
from threading import RLock
from typing import Any, Generic, TypeVar

T = TypeVar("T")

BUILTIN_DISTRIBUTION_NAME = "evaluation-studio"

ENTRY_POINT_GROUPS: tuple[str, ...] = (
    "eval_studio.models",
    "eval_studio.solvers",
    "eval_studio.filters",
    "eval_studio.metrics",
    "eval_studio.aggregations",
    "eval_studio.sandboxes",
)

_BUILTIN_SPECS: dict[str, dict[str, str]] = {
    "eval_studio.models": {
        "mock": "eval_studio.gateway.mock:MockGateway",
        "http-byok": "eval_studio.gateway.http:HttpBYOKGateway",
        "hf": "eval_studio.gateway.adapters.hf:HuggingFaceAdapter",
        "vllm": "eval_studio.gateway.adapters.vllm:VLLMAdapter",
    },
    "eval_studio.solvers": {},
    "eval_studio.filters": {
        "multi_choice_regex": "eval_studio.scoring.filters:MultiChoiceRegexFilter",
        "regex": "eval_studio.scoring.filters:RegexFilter",
        "take_first": "eval_studio.scoring.filters:TakeFirstFilter",
        "take_first_k": "eval_studio.scoring.filters:TakeFirstKFilter",
        "majority_vote": "eval_studio.scoring.filters:MajorityVoteFilter",
        "think_strip": "eval_studio.scoring.filters:ThinkStripFilter",
    },
    "eval_studio.metrics": {
        "acc": "eval_studio.scoring.metrics:acc",
        "acc_norm": "eval_studio.scoring.metrics:acc_norm",
        "acc_bytes": "eval_studio.scoring.metrics:acc_bytes",
        "acc_mutual_info": "eval_studio.scoring.metrics:acc_mutual_info",
        "acc_all": "eval_studio.scoring.metrics:acc_all",
        "brier_score": "eval_studio.scoring.metrics:brier_score",
        "bits_per_byte": "eval_studio.scoring.metrics:bits_per_byte",
        "word_perplexity": "eval_studio.scoring.metrics:word_perplexity",
        "byte_perplexity": "eval_studio.scoring.metrics:byte_perplexity",
        "weighted_perplexity": "eval_studio.scoring.metrics:weighted_perplexity",
        "bypass": "eval_studio.scoring.metrics:bypass",
        "exact_match": "eval_studio.scoring.metrics:exact_match",
        "numeric_exact_match": "eval_studio.scoring.metrics:numeric_exact_match",
        "whole_word_match": "eval_studio.scoring.metrics:whole_word_match",
        "latex_exact_match": "eval_studio.scoring.metrics:latex_exact_match",
        "classification_f1": "eval_studio.scoring.metrics:classification_f1",
        "f1": "eval_studio.scoring.metrics:classification_f1",
        "token_f1": "eval_studio.scoring.metrics:token_f1",
        "squad_f1": "eval_studio.scoring.metrics:token_f1",
    },
    "eval_studio.aggregations": {
        "mean": "eval_studio.stats.aggregation:agg_mean",
        "mean_stderr": "eval_studio.stats.bootstrap:mean_stderr",
        "bootstrap_stderr": "eval_studio.stats.bootstrap:bootstrap_stderr",
        "pooled_sample_stderr": "eval_studio.stats.aggregation:agg_pooled_sample_stderr",
        "unweighted_mean_stderr": "eval_studio.stats.aggregation:agg_unweighted_mean_stderr",
        "aggregate_group": "eval_studio.stats.aggregation:aggregate_group",
        "clustered_stderr": "eval_studio.stats.aggregation:clustered_stderr",
        "weighted_perplexity": "eval_studio.stats.aggregation:agg_weighted_perplexity",
        "bits_per_byte": "eval_studio.stats.aggregation:agg_bits_per_byte",
        "brier_score": "eval_studio.stats.aggregation:agg_brier_score",
        "bypass": "eval_studio.stats.aggregation:agg_bypass",
    },
    "eval_studio.sandboxes": {
        "docker": "eval_studio.sandbox.docker:DockerSandbox",
    },
}


class PluginLoadError(RuntimeError):
    """Raised when loading custom local/external plugin modules fails or is untrusted."""


def _is_builtin_entry_point(ep: importlib_metadata.EntryPoint) -> bool:
    dist = getattr(ep, "dist", None)
    if dist is not None:
        return getattr(dist, "name", None) == BUILTIN_DISTRIBUTION_NAME
    return ep.value.startswith("eval_studio.")


def _load_target_string(target: str) -> Any:
    module_name, _, attr_path = target.partition(":")
    if not module_name or not attr_path:
        raise PluginLoadError(f"Invalid plugin target specification: '{target}'")
    obj: Any = importlib.import_module(module_name)
    for part in attr_path.split("."):
        obj = getattr(obj, part)
    return obj


class PluginRegistry(Generic[T]):
    """Lazy registry storing entry-point placeholders until first access.

    Colliding external plugin names never shadow built-in registrations.
    """

    def __init__(self, group: str) -> None:
        self.group = group
        self._lock = RLock()
        self._builtins: dict[str, T] = {}
        self._lazy_builtins: dict[str, importlib_metadata.EntryPoint | str] = {}
        self._lazy_entry_points: dict[str, importlib_metadata.EntryPoint] = {}
        self._resolved: dict[str, T] = {}
        self._discovered = False

    def register_lazy_builtin(self, name: str, target: str) -> None:
        """Seed a built-in lazy 'module:attr' placeholder without importing it."""
        with self._lock:
            if name not in self._builtins and name not in self._lazy_builtins:
                self._lazy_builtins[name] = target

    def is_builtin(self, name: str) -> bool:
        """Return True if `name` is registered (eagerly or lazily) as a built-in."""
        with self._lock:
            return name in self._builtins or name in self._lazy_builtins

    def register(self, name: str, obj: T, *, builtin: bool = True) -> None:
        """Register a component directly (external plugins never shadow built-ins)."""
        with self._lock:
            if builtin:
                self._builtins[name] = obj
                self._resolved[name] = obj
            elif (
                name not in self._builtins
                and name not in self._lazy_builtins
                and name not in self._resolved
            ):
                self._resolved[name] = obj

    def discover(self) -> None:
        """Lazily index importlib.metadata entry points without importing them."""
        with self._lock:
            if self._discovered:
                return
            eps = importlib_metadata.entry_points(group=self.group)
            builtin_eps: list[importlib_metadata.EntryPoint] = []
            external_eps: list[importlib_metadata.EntryPoint] = []
            for ep in eps:
                if _is_builtin_entry_point(ep):
                    builtin_eps.append(ep)
                else:
                    external_eps.append(ep)

            # Index built-in entry points first so third-party packages can never shadow them.
            for ep in builtin_eps:
                if ep.name not in self._builtins and ep.name not in self._lazy_builtins:
                    self._lazy_builtins[ep.name] = ep

            for ep in external_eps:
                if (
                    ep.name in self._builtins
                    or ep.name in self._lazy_builtins
                    or ep.name in self._lazy_entry_points
                ):
                    continue
                self._lazy_entry_points[ep.name] = ep

            self._discovered = True

    def list_names(self) -> list[str]:
        """Return all registered and lazily discovered plugin names."""
        self.discover()
        with self._lock:
            return sorted(
                set(self._builtins)
                | set(self._lazy_builtins)
                | set(self._resolved)
                | set(self._lazy_entry_points)
            )

    def get(self, name: str) -> T:
        """Resolve and return a registered component by name."""
        self.discover()
        with self._lock:
            if name in self._resolved:
                return self._resolved[name]
            if name in self._lazy_builtins:
                spec = self._lazy_builtins[name]
                loaded_builtin: T = (
                    _load_target_string(spec) if isinstance(spec, str) else spec.load()
                )
                self._builtins[name] = loaded_builtin
                self._resolved[name] = loaded_builtin
                return loaded_builtin
            if name in self._lazy_entry_points:
                loaded: T = self._lazy_entry_points[name].load()
                self._resolved[name] = loaded
                return loaded
        raise KeyError(f"Unknown plugin '{name}' in group '{self.group}'")


def _create_registries() -> dict[str, PluginRegistry[Any]]:
    registries: dict[str, PluginRegistry[Any]] = {}
    for group in ENTRY_POINT_GROUPS:
        reg: PluginRegistry[Any] = PluginRegistry(group)
        for name, target in _BUILTIN_SPECS.get(group, {}).items():
            reg.register_lazy_builtin(name, target)
        registries[group] = reg
    return registries


_REGISTRIES: dict[str, PluginRegistry[Any]] = _create_registries()


def get_registry(group: str) -> PluginRegistry[Any]:
    """Return the singleton lazy registry for a plugin entry-point group."""
    if group not in _REGISTRIES:
        raise ValueError(
            f"Unsupported plugin group '{group}'. Expected one of {ENTRY_POINT_GROUPS}"
        )
    return _REGISTRIES[group]


def load_plugins(
    modules: Sequence[str],
    *,
    trust_remote_code: bool = False,
) -> list[str]:
    """Import custom plugin modules or .py paths after verifying --trust-remote-code."""
    if not modules:
        return []
    if not trust_remote_code:
        raise PluginLoadError(
            "Loading custom plugins via --plugins requires explicit --trust-remote-code opt-in."
        )

    loaded_names: list[str] = []
    for entry in modules:
        entry_path = Path(entry)
        if entry.endswith(".py") or entry_path.is_file():
            if not entry_path.is_file():
                raise PluginLoadError(f"Plugin file not found: '{entry}'")
            module_name = f"eval_studio_user_plugin_{entry_path.stem}"
            spec = importlib.util.spec_from_file_location(module_name, entry_path)
            if spec is None or spec.loader is None:
                raise PluginLoadError(f"Could not create module spec for plugin '{entry}'")
            mod = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = mod
            try:
                spec.loader.exec_module(mod)
            except Exception as exc:
                sys.modules.pop(module_name, None)
                raise PluginLoadError(f"Failed executing plugin '{entry}': {exc}") from exc
            loaded_label = str(entry_path)
        else:
            try:
                mod = importlib.import_module(entry)
            except Exception as exc:
                raise PluginLoadError(f"Failed importing plugin module '{entry}': {exc}") from exc
            loaded_label = mod.__name__

        register_fn = getattr(mod, "register_plugins", None)
        if callable(register_fn):
            register_fn(get_registry)

        plugin_map = getattr(mod, "EVAL_STUDIO_PLUGINS", None)
        if isinstance(plugin_map, Mapping):
            for group, entries in plugin_map.items():
                if not isinstance(group, str) or not isinstance(entries, Mapping):
                    raise PluginLoadError(
                        f"Invalid EVAL_STUDIO_PLUGINS structure in plugin '{loaded_label}'."
                    )
                registry = get_registry(group)
                for plugin_name, obj in entries.items():
                    registry.register(str(plugin_name), obj, builtin=False)

        loaded_names.append(loaded_label)

    return loaded_names
