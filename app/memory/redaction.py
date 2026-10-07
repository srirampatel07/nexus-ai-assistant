"""Secret detection/redaction for NEXUS memory (Phase 3).

Independent layer: deliberate duplicate of the *spirit* of the audit
sanitizer, but memory has stricter rules — it must REFUSE predominantly
secret material instead of storing it. Never imports audit.py.

Policy:
- `reject` — content is (mostly) secret material: private keys, raw
  tokens/keys, credential assignments, `.env`-style secret lines.
  The memory is NOT stored at all.
- `redact` — secrets embedded in longer prose: values replaced with `***`
  and the redacted text may be stored.
- `keep` — no secrets detected.

Placeholder values (`your-...`, `xxx`, `test`, `<key>`, ...) never trigger.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_KEEP = "keep"
_REDACT = "redact"
_REJECT = "reject"

_PLACEHOLDERS = {
    "test", "testing", "example", "xxx", "***", "...", "null", "none",
    "fake", "dummy", "placeholder", "changeme", "password", "secret",
}

_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
    re.DOTALL,
)
_BEARER_RE = re.compile(r"Bearer\s+([A-Za-z0-9\-._~+/=]{8,})", re.IGNORECASE)
_PREFIXED_TOKEN_RE = re.compile(r"\b((?:gsk_|sk-|xox[bap]-)[A-Za-z0-9\-_]{8,})\b")
# KEY = value / password: value style assignments (incl. .env lines).
_ASSIGNMENT_RE = re.compile(
    r"(?i)\b([A-Z0-9_]*(?:API[_-]?KEY|PASSWORD|PASSWD|PWD|SECRET|TOKEN|CREDENTIAL|AUTH)[A-Z0-9_]*)\s*[:=]\s*(\S+)"
)
# Quoted assignments: "password": "hunter2".
_QUOTED_ASSIGNMENT_RE = re.compile(
    r'(?i)"([A-Za-z0-9_]*(?:password|passwd|secret|token|api[_-]?key)[A-Za-z0-9_]*)"\s*:\s*"([^"]+)"'
)
_ENV_LINE_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(\S+)\s*$")
_SECRET_NAME_RE = re.compile(
    r"(?i)(KEY|TOKEN|SECRET|PASSWORD|PASSWD|PWD|CREDENTIAL|AUTH)"
)


def _is_placeholder(value: str) -> bool:
    v = value.strip().strip("\"'").lower()
    if not v or v in _PLACEHOLDERS:
        return True
    return v.startswith(("your-", "your_", "<", "example", "test-", "fake-"))


def _has_mixed_classes(token: str) -> bool:
    classes = sum(
        (
            any(c.islower() for c in token),
            any(c.isupper() for c in token),
            any(c.isdigit() for c in token),
            any(not c.isalnum() for c in token),
        )
    )
    return classes >= 2


@dataclass
class RedactionResult:
    decision: str  # keep | redact | reject
    text: str
    reasons: list[str] = field(default_factory=list)


def evaluate(content: str) -> RedactionResult:
    """Inspect content for secrets. Never raises; never returns secrets."""
    text = content or ""
    stripped = text.strip()
    reasons: list[str] = []

    if not stripped:
        return RedactionResult(decision=_KEEP, text=text)

    # 1. Private key material anywhere -> always reject.
    if _PRIVATE_KEY_RE.search(text):
        return RedactionResult(
            decision=_REJECT, text="", reasons=["private key material"]
        )

    lines = [ln for ln in text.splitlines() if ln.strip()]
    single_line = len(lines) <= 1

    def _lone_credential(line: str) -> str | None:
        """Detect a credential token that dominates a single line.

        Returns a reason, or None. A bare `Bearer <tok>` / prefixed token
        accompanied by at most a short label (e.g. `Authorization:`) is a
        credential statement -> reject. Tokens buried in real prose fall
        through to redaction.
        """
        for pattern in (_BEARER_RE, _PREFIXED_TOKEN_RE):
            match = pattern.search(line)
            if not match or _is_placeholder(match.group(1)):
                continue
            rest = pattern.sub("", line)
            words = [w for w in re.split(r"[\s:,;]+", rest) if w]
            if len(words) <= 2:
                return "credential statement"
        return None

    def secret_line_count() -> int:
        count = 0
        for ln in lines:
            s = ln.strip()
            if _BEARER_RE.search(s) and not _is_placeholder(_BEARER_RE.search(s).group(1)):  # type: ignore[union-attr]
                count += 1
                continue
            m = _PREFIXED_TOKEN_RE.search(s)
            if m and not _is_placeholder(m.group(1)):
                count += 1
                continue
            m = _ASSIGNMENT_RE.search(s)
            if m and not _is_placeholder(m.group(2)):
                count += 1
                continue
            m = _ENV_LINE_RE.match(s)
            if (
                m
                and _SECRET_NAME_RE.search(m.group(1))
                and not _is_placeholder(m.group(2))
            ):
                count += 1
                continue
        return count

    # 2. Predominantly secret material -> reject, never store.
    if single_line:
        s = stripped
        lone = _lone_credential(s)
        if lone:
            return RedactionResult(decision=_REJECT, text="", reasons=[lone])
        m = _ASSIGNMENT_RE.fullmatch(s) or _QUOTED_ASSIGNMENT_RE.fullmatch(s)
        if m and not _is_placeholder(m.group(2)):
            return RedactionResult(
                decision=_REJECT, text="", reasons=["credential assignment"]
            )
        m = _ENV_LINE_RE.match(s)
        if (
            m
            and _SECRET_NAME_RE.search(m.group(1))
            and not _is_placeholder(m.group(2))
        ):
            return RedactionResult(
                decision=_REJECT, text="", reasons=["secret assignment"]
            )
        # Bare high-entropy token with no label (e.g. pasted key).
        token = s.strip("\"'")
        if len(token) >= 20 and " " not in token and _has_mixed_classes(token):
            return RedactionResult(
                decision=_REJECT, text="", reasons=["raw secret value"]
            )
    else:
        if secret_line_count() * 2 >= len(lines):
            return RedactionResult(
                decision=_REJECT, text="", reasons=["mostly secret material"]
            )

    # 3. Incidental embeddings -> redact values, keep prose.
    redacted = text

    def _sub(pattern: re.Pattern[str], source: str, group: int = 1) -> str:
        def repl(match: re.Match[str]) -> str:
            return (
                match.group(0).replace(match.group(group), "***", 1)
                if not _is_placeholder(match.group(group))
                else match.group(0)
            )

        return pattern.sub(repl, source)

    redacted = _sub(_BEARER_RE, redacted)
    redacted = _sub(_PREFIXED_TOKEN_RE, redacted)
    redacted = _sub(_ASSIGNMENT_RE, redacted, 2)
    redacted = _sub(_QUOTED_ASSIGNMENT_RE, redacted, 2)

    if redacted != text:
        reasons.append("embedded secrets redacted")
        return RedactionResult(decision=_REDACT, text=redacted, reasons=reasons)
    return RedactionResult(decision=_KEEP, text=text)
