"""Shared pytest fixtures (Phase 1). No real credentials or network."""

import pytest

from app.brain.provider import EchoProvider
from app.core.config import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(
        ai_provider="echo",
        ai_model="test-model",
        max_history_messages=10,
    )


@pytest.fixture
def echo_provider() -> EchoProvider:
    return EchoProvider()
