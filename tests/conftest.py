"""Shared pytest fixtures and network kill-switch."""

from __future__ import annotations

import socket
from collections.abc import Generator
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Directory containing committed test fixtures (Wikipedia HTML, etc.)."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def sp500_wikipedia_html(fixtures_dir: Path) -> str:
    """Real Wikipedia 'List of S&P 500 companies' HTML, captured 2026-05-05."""
    return (fixtures_dir / "sp500_wikipedia.html").read_text(encoding="utf-8")


@pytest.fixture(autouse=True)
def block_network(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Generator[None, None, None]:
    """Hard-block sockets in any test not marked ``@pytest.mark.network``
    or ``@pytest.mark.mock_http``.

    The ``mock_http`` exemption exists because asyncio's event loop creates
    internal sockets even when ``httpx.MockTransport`` intercepts all HTTP
    traffic — those tests don't reach the wire but the runtime still needs
    socket() to work for its self-pipe.

    Live-network tests opt in explicitly via ``@pytest.mark.network``.
    """
    if request.node.get_closest_marker("network") or request.node.get_closest_marker("mock_http"):
        yield
        return

    real_socket = socket.socket

    def guarded(*args: object, **kwargs: object) -> socket.socket:
        raise RuntimeError(
            "network access blocked in unit tests. "
            "Use a fixture or mark the test @pytest.mark.network "
            "(real network) or @pytest.mark.mock_http (httpx MockTransport) to opt in."
        )

    monkeypatch.setattr(socket, "socket", guarded)
    yield
    monkeypatch.setattr(socket, "socket", real_socket)
