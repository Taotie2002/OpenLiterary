"""Minimal in-memory ``typesafe_sdk`` stub for optional-dependency tests.

Round 145's batching/identity tests exercise pure Python control flow and do
not need the real HTTP SDK.  Using a stub keeps them runnable on Python 3.10
deployments where the optional System One dependency is not installed.
"""
from __future__ import annotations

import importlib.machinery
import sys
import types


class _Question:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeTypeSafeClient:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def install(monkeypatch):
    """Install a fake ``typesafe_sdk`` module into ``sys.modules``."""
    module = types.ModuleType("typesafe_sdk")
    module.__spec__ = importlib.machinery.ModuleSpec("typesafe_sdk", loader=None)
    module.Choice = _Question
    module.Score = _Question
    module.Noul = _Question
    module.TypeSafeClient = FakeTypeSafeClient
    monkeypatch.setitem(sys.modules, "typesafe_sdk", module)
    return module
