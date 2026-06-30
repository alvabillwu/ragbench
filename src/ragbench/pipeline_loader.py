"""Pluggable user-pipeline loader.

Lets a user benchmark their OWN retriever/generator without forking ragbench:

    # mypipe.py
    from ragbench.types import RetrievedDoc, Answer

    def retriever(query: str) -> list[RetrievedDoc]:
        ...

    def generator(query: str, docs: list[RetrievedDoc]) -> Answer:
        ...

Then:

    ragbench run --pipeline mypipe.py --dataset factual

The loader imports the module from a filesystem path and pulls out the
`retriever` and `generator` callables, with clear errors if they're missing or
the wrong shape. This is what makes ragbench a real benchmarking *tool* rather
than a self-referential demo.
"""

from __future__ import annotations

import importlib.util
import inspect
import os
import sys
from pathlib import Path
from typing import Optional

from .types import RetrievedDoc, Answer, RetrieverFn, GeneratorFn


class PipelineLoadError(Exception):
    """Raised when a user pipeline module can't be loaded or is malformed."""


def _load_module(path: str | Path):
    """Import a Python module from an arbitrary file path."""
    p = Path(path).resolve()
    if not p.exists():
        raise PipelineLoadError(f"pipeline file not found: {p}")
    if p.suffix != ".py":
        raise PipelineLoadError(f"pipeline file must be a .py file: {p}")
    # Use a stable module name derived from the filename so re-imports don't
    # collide; prefix to avoid shadowing real packages.
    mod_name = f"ragbench_user_pipeline_{p.stem}"
    spec = importlib.util.spec_from_file_location(mod_name, p)
    if spec is None or spec.loader is None:
        raise PipelineLoadError(f"could not build module spec for {p}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise PipelineLoadError(f"error importing {p}: {e}") from e
    return module


def _check_callable(obj, name: str, module_name: str) -> None:
    if not hasattr(obj, "__call__"):
        raise PipelineLoadError(
            f"`{name}` in {module_name} is not callable (got {type(obj).__name__})"
        )


def load_pipeline(path: str | Path) -> tuple[RetrieverFn, GeneratorFn]:
    """Load (retriever, generator) callables from a user pipeline module.

    The module must expose top-level `retriever` and `generator` callables.
    """
    module = _load_module(path)
    module_name = getattr(module, "__file__", str(path))
    missing = [n for n in ("retriever", "generator") if not hasattr(module, n)]
    if missing:
        raise PipelineLoadError(
            f"pipeline module {module_name} is missing required export(s): {', '.join(missing)}. "
            f"Define top-level `retriever(query: str) -> list[RetrievedDoc]` and "
            f"`generator(query: str, docs: list[RetrievedDoc]) -> Answer`."
        )
    retriever = module.retriever
    generator = module.generator
    _check_callable(retriever, "retriever", module_name)
    _check_callable(generator, "generator", module_name)
    return retriever, generator
