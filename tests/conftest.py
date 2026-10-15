"""Shared fixtures: one small bootstrapped platform per test session in a temporary directory."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("RS_SUPERVISOR_INTERVAL", "0")
os.environ.setdefault("RS_BACKGROUND_TRAFFIC", "false")
os.environ.pop("GEMINI_API_KEY", None)  # tests exercise the offline engines; Gemini paths use a fake client


@pytest.fixture(scope="session")
def platform(tmp_path_factory):
    from railsentinel.config import load_settings
    from railsentinel.core import Platform

    os.environ["RS_DATA_DIR"] = str(tmp_path_factory.mktemp("railsentinel_data"))
    p = Platform(load_settings())
    p.bootstrap(scale=0.35, warmup_batches=2, progress=lambda *_: None)
    Platform.set_instance(p)
    yield p
    Platform.set_instance(None)


@pytest.fixture()
def clean(platform):
    """Return the platform to its launch state before and after a test that changes the world."""
    platform.chaos.reset(actor="test")
    yield platform
    platform.chaos.reset(actor="test")
