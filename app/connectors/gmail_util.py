"""Shared Gmail helpers: credential + body-text extraction (no secrets logged)."""

from __future__ import annotations

import os
import re


def creds() -> tuple[str, str]:
    """App passwords are shown with spaces; IMAP wants them without."""
    return (os.environ.get("GMAIL_ADDRESS", "").strip(),
            "".join(os.environ.get("GMAIL_APP_PASSWORD", "").split()))


def decode_header_value(value: str, limit: int = 200) -> str:
    from email.header import decode_header
    out = ""
    for chunk, enc in decode_header(value or ""):
        out += chunk.decode(enc or "utf-8", "replace") if isinstance(chunk, bytes) else chunk
    return out[:limit]


def body_text(msg, limit: int = 8000) -> str:
    """Best plain-text body: text/plain parts first, else stripped HTML."""
    chunks: list[str] = []
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_maintype() == "multipart":
                continue
            if part.get("Content-Disposition", "").strip().lower().startswith("attachment"):
                continue
            ctype = (part.get_content_type() or "").lower()
            if ctype == "text/plain":
                try:
                    chunks.append(part.get_payload(decode=True).decode(
                        part.get_content_charset() or "utf-8", "replace"))
                except Exception:
                    pass
        if not chunks:
            for part in msg.walk():
                if (part.get_content_type() or "").lower() == "text/html":
                    try:
                        html = part.get_payload(decode=True).decode(
                            part.get_content_charset() or "utf-8", "replace")
                        chunks.append(re.sub(r"<[^>]+>", " ", html))
                    except Exception:
                        pass
                    break
    else:
        try:
            payload = msg.get_payload(decode=True)
            if isinstance(payload, bytes):
                chunks.append(payload.decode(msg.get_content_charset() or "utf-8", "replace"))
            elif isinstance(payload, str):
                chunks.append(payload)
        except Exception:
            pass
    text = re.sub(r"[ \t]+", " ", " ".join(chunks))
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text[:limit]
