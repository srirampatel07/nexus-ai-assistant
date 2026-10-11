"""Visible Brave browser launcher (Phase 6).

`brave.open` opens any valid public HTTP/HTTPS URL in the user's real,
visible Brave desktop browser on Windows — a separate operation from the
internal reader sessions (`browser.*`), which never open desktop windows.

- No domain allowlist: any valid HTTP/HTTPS destination is accepted,
  including domains NEXUS has never seen. Destinations are never invented:
  the target is either a full URL or a reference ("result 2", "second",
  "first", "last") resolved against the current `SearchSession` of real
  `web.search` output.
- Client-side navigation is NOT server-side fetching: SSRF protections
  stay on `web.fetch`/`browser.*`, while this tool applies URL validation
  only (http/https scheme, hostname present, no embedded credentials,
  unsafe schemes such as file:/javascript:/data: refused).
- LOW risk, no confirmation: opening a validated URL is a minor,
  user-visible side effect (like launching an allowlisted app). Validation
  still runs first — invalid targets are rejected before any launch.
- The result reports that Brave was *started* (with pid) — never that the
  page loaded, which NEXUS cannot verify. Launch uses argv (shell=False);
  no shell ever sees the URL.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import BaseModel, Field

from app.core.exceptions import ToolDenied
from app.security.policy import PermissionLevel
from app.tools.base import Tool, ToolResult
from app.web.search_session import SearchSession

_MAX_URL_CHARS = 2048

_UNSAFE_SCHEMES = frozenset(
    {
        "file",
        "javascript",
        "data",
        "vbscript",
        "ftp",
        "jar",
        "about",
        "chrome",
        "edge",
        "brave",
        "view-source",
    }
)

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x20\x7f]")


def validate_desktop_url(raw: str) -> str:
    """Validate a desktop-browser destination URL.

    Returns the stripped original URL (path/query/fragment preserved
    exactly). Raises ToolDenied for anything that is not a plain
    HTTP/HTTPS URL without embedded credentials.
    """
    text = (raw or "").strip()
    if not text:
        raise ToolDenied("Empty URL: provide a full http(s) URL to open.")
    if len(text) > _MAX_URL_CHARS:
        raise ToolDenied(
            f"URL is too long ({len(text)} chars > {_MAX_URL_CHARS})."
        )
    try:
        parts = urlsplit(text)
    except ValueError as exc:
        raise ToolDenied(f"Malformed URL: {exc}") from exc
    scheme = parts.scheme.lower()
    if not scheme:
        raise ToolDenied(
            f"URL {text!r} has no scheme; use a full URL "
            "starting with http:// or https://."
        )
    if scheme in _UNSAFE_SCHEMES or scheme not in ("http", "https"):
        raise ToolDenied(
            f"Refused URL scheme {scheme!r}: only http and https "
            "may be opened in the desktop browser."
        )
    if parts.username or parts.password or "@" in (parts.netloc or ""):
        raise ToolDenied("Refused URL: credentials embedded in URLs are never opened.")
    if not parts.hostname:
        raise ToolDenied(f"URL {text!r} has no hostname.")
    if _CONTROL_CHARS_RE.search(text):
        raise ToolDenied("Refused URL: it contains whitespace or control characters.")
    return text


def normalize_bare_host(raw: str) -> str | None:
    """Turn a bare `example.com/path` into an https URL, else None."""
    text = (raw or "").strip()
    if not text or " " in text or "://" in text:
        return None
    if not re.fullmatch(r"[A-Za-z0-9_.-]+(?::\d+)?(?:/[^\\]*)?", text):
        return None
    host = text.split("/")[0].split(":")[0]
    if "." not in host or host.startswith((".", "-")) or host.endswith((".", "-")):
        return None
    return f"https://{text}"


def find_brave_executable(
    explicit: str | None = None,
    path_checker: Callable[[str], bool] | None = None,
) -> str | None:
    """Locate brave.exe via registered locations + standard install paths.

    Order: explicit override -> Windows App Paths registry -> standard
    per-machine/per-user install directories -> PATH lookup. Returns None
    when Brave is not found. `path_checker` is injectable for tests.
    """
    exists = path_checker or os.path.isfile
    if explicit and exists(explicit):
        return explicit

    # Registered application location (Windows App Paths).
    try:
        import winreg  # type: ignore[import-not-found]

        for root in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
            try:
                with winreg.OpenKey(
                    root, r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\brave.exe"
                ) as key:
                    registered, _ = winreg.QueryValueEx(key, "")
                    if registered and exists(str(registered)):
                        return str(registered)
            except OSError:
                continue
    except ImportError:
        pass

    candidates: list[str] = []
    for env_var in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"):
        base = os.environ.get(env_var)
        if base:
            candidates.append(
                str(
                    Path(base)
                    / "BraveSoftware"
                    / "Brave-Browser"
                    / "Application"
                    / "brave.exe"
                )
            )
    for candidate in candidates:
        if exists(candidate):
            return candidate

    found = shutil.which("brave") or shutil.which("brave.exe")
    if found and exists(found):
        return found
    return None


class BraveOpenInput(BaseModel):
    target: str = Field(
        description=(
            "Full http(s) URL to open (path/query/fragment preserved), "
            "a bare hostname like example.com/docs, or a search-result "
            "reference such as 'result 2', 'second', 'first', or 'last'."
        ),
        min_length=1,
        max_length=_MAX_URL_CHARS + 32,
    )


class BraveOpenTool(Tool):
    """Open any valid HTTP/HTTPS destination in the visible Brave browser."""

    name = "brave.open"
    description = (
        "Open a web destination in the user's visible Brave desktop browser. "
        "Pass a full http(s) URL (any public domain, path/query kept), a bare "
        "hostname, or a current-search reference like 'result 2'/'second'/"
        "'first'/'last'. Reports the launch honestly."
    )
    category = "browser"
    risk_level = PermissionLevel.LOW
    requires_confirmation = False
    input_model = BraveOpenInput

    def __init__(
        self,
        session: SearchSession | None = None,
        brave_exe: str | None = None,
        popen: Callable[..., object] | None = None,
        finder: Callable[[str | None], str | None] | None = None,
    ) -> None:
        # NB: `or` is wrong here — an empty session is falsy via __len__.
        self.session = session if session is not None else SearchSession()
        self._brave_exe = brave_exe
        self._popen = popen or subprocess.Popen
        self._finder = finder or find_brave_executable

    def _resolve_target(self, target: str) -> str:
        """Return the exact URL to open, or raise ToolDenied with guidance."""
        text = (target or "").strip()
        if not text:
            raise ToolDenied("Empty target: provide a URL or a result reference.")
        # 1. Search-result reference against real search output.
        resolved = self.session.resolve(text)
        if resolved is not None:
            return validate_desktop_url(resolved)
        # 2. Full URL as given (validated, preserved exactly).
        try:
            return validate_desktop_url(text)
        except ToolDenied:
            pass
        # 3. Bare hostname ("open python.org") -> https.
        bare = normalize_bare_host(text)
        if bare is not None:
            return validate_desktop_url(bare)
        # 4. Not a reference (empty session?) vs genuinely unresolvable.
        if len(self.session) == 0 and not urlsplit(text).scheme:
            raise ToolDenied(
                f"Could not resolve {text!r}: there are no search results in "
                "the current session yet. Search first, then open a result "
                "by number, or pass a full http(s) URL."
            )
        raise ToolDenied(
            f"Could not resolve {text!r} to a web destination. Pass a full "
            "http(s) URL or a current-search reference like 'result 2'."
        )

    def assess(self, validated: BaseModel) -> PermissionLevel:
        assert isinstance(validated, BraveOpenInput)
        # Syntax-level policy gate (audited deny); reference resolution and
        # Brave discovery happen in execute() with honest failure results.
        self._resolve_target(validated.target)
        return self.risk_level

    def execute(self, validated: BaseModel) -> ToolResult:
        assert isinstance(validated, BraveOpenInput)
        try:
            url = self._resolve_target(validated.target)
        except ToolDenied as exc:
            return ToolResult(ok=False, error=str(exc))
        exe = self._finder(self._brave_exe)
        if not exe:
            return ToolResult(
                ok=False,
                error=(
                    "Brave browser was not found in the standard install "
                    "locations or PATH. Install Brave (or set its path) and "
                    "try again. Nothing was opened."
                ),
            )
        try:
            proc = self._popen(  # type: ignore[operator]
                [exe, "--new-window", url],
                shell=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            return ToolResult(ok=False, error=f"Could not start Brave: {exc}")
        except Exception as exc:  # never leak internals
            return ToolResult(
                ok=False, error=f"Brave launch failed: {type(exc).__name__}."
            )
        pid = getattr(proc, "pid", None)
        return ToolResult(
            ok=True,
            output={
                "url": url,
                "browser": "brave",
                "executable": exe,
                "pid": pid,
                "launched": True,
                "note": (
                    "Brave was started with the URL; page load was not "
                    "verified. Confirm visually in the Brave window. "
                    "The page is now visible to the user there — do not "
                    "call web.fetch or browser.* tools to re-read this URL "
                    "unless the user explicitly asked for its content. "
                    "Reply directly with the opened URL."
                ),
            },
        )
