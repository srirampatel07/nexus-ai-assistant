"""SSRF protection for NEXUS web access (Phase 6).

Every outbound URL — initial and each redirect hop — passes through
`check_url()` BEFORE any connection is opened:

  scheme allowlist -> no userinfo -> hostname present and dotted ->
  not a local/special name -> IP literals must be globally routable ->
  DNS must resolve, and EVERY resolved address must be globally routable.

Fail-closed: unresolvable hosts are refused. DNS resolution is injectable
(`dns_resolver`) so tests never touch real DNS.
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Callable, Sequence
from urllib.parse import urlsplit

from app.web.errors import UnsafeDestinationError

# Names that never leave the machine / never belong to public content.
_LOCAL_NAMES = {"localhost"}
# Suffixes reserved for local, special-use, documentation, or test domains.
_LOCAL_SUFFIXES = (
    ".localhost",
    ".local",
    ".internal",
    ".invalid",
    ".example",
    ".test",
    ".home",
    ".lan",
    ".corp",
    ".localdomain",
)

# Well-known cloud metadata endpoints (also link-local, belt and braces).
_METADATA_HOSTS = {
    "169.254.169.254",
    "metadata.google.internal",
    "metadata.goog",
}

AddrInfo = tuple


def default_resolver(host: str) -> Sequence[AddrInfo]:
    """Resolve `host` via the OS resolver (stream sockets only)."""
    return socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)


def _is_routable_ip(value: str) -> bool:
    """True only for globally routable addresses (no private/loopback/etc)."""
    try:
        addr = ipaddress.ip_address(value)
    except ValueError:
        return False
    return addr.is_global and not addr.is_multicast


def check_url(
    raw: str,
    *,
    dns_resolver: Callable[[str], Sequence[AddrInfo]] = default_resolver,
) -> str:
    """Validate `raw` as a safe outbound HTTP(S) URL.

    Returns the unchanged URL when safe. Raises UnsafeDestinationError
    (with host-only diagnostics, never credentials or bodies) otherwise.
    """
    text = (raw or "").strip()
    if not text:
        raise UnsafeDestinationError("Empty URL is not allowed.")
    try:
        parts = urlsplit(text)
    except ValueError as exc:
        raise UnsafeDestinationError("URL is malformed.") from exc
    if parts.scheme.lower() not in ("http", "https"):
        raise UnsafeDestinationError(
            f"URL scheme '{parts.scheme or '(none)'}' is not allowed "
            "(only http/https)."
        )
    if parts.username or parts.password:
        raise UnsafeDestinationError("URLs with credentials are not allowed.")
    host = (parts.hostname or "").lower()
    if not host:
        raise UnsafeDestinationError("URL has no host.")
    if host in _LOCAL_NAMES or host in _METADATA_HOSTS:
        raise UnsafeDestinationError(f"Host '{host}' is restricted.")
    if host.endswith(_LOCAL_SUFFIXES):
        raise UnsafeDestinationError(f"Host '{host}' is restricted.")
    if "." not in host and ":" not in host:
        # Single-label names resolve via local search domains — intranet.
        raise UnsafeDestinationError(f"Host '{host}' is restricted.")
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None and (literal.is_multicast or not literal.is_global):
        raise UnsafeDestinationError(f"Host '{host}' is restricted.")
    if literal is None:
        try:
            answers = dns_resolver(host)
        except OSError as exc:
            raise UnsafeDestinationError(
                f"Host '{host}' could not be resolved; refused."
            ) from exc
        if not answers:
            raise UnsafeDestinationError(
                f"Host '{host}' could not be resolved; refused."
            )
        for answer in answers:
            try:
                addr = answer[4][0]
            except (IndexError, TypeError):
                raise UnsafeDestinationError(
                    f"Host '{host}' resolved unexpectedly; refused."
                )
            if not _is_routable_ip(str(addr)):
                raise UnsafeDestinationError(f"Host '{host}' is restricted.")
    return text
