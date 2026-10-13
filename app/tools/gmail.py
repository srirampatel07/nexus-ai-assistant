"""Gmail tools via IMAP/SMTP app password (no OAuth dance needed).

Credentials come from env only: GMAIL_ADDRESS + GMAIL_APP_PASSWORD
(Google Account -> 2-Step -> App passwords). Values never logged.
Dashboard /api/connectors/gmail persists them to .env on request.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.connectors.gmail_util import body_text, creds, decode_header_value
from app.security.policy import PermissionLevel
from app.tools.base import Tool, ToolResult


class GmailInboxInput(BaseModel):
    limit: int = Field(default=10, ge=1, le=25)


class GmailSendInput(BaseModel):
    to: str = Field(min_length=3, max_length=320)
    subject: str = Field(min_length=1, max_length=200)
    body: str = Field(min_length=1, max_length=10000)
    in_reply_to: str = Field(default="", max_length=500)
    references: str = Field(default="", max_length=1000)


def compose_reply(original: dict, reply_text: str) -> dict:
    """Build human-style reply fields: sender, Re: subject, threading
    headers — and ONLY the user's own words as the body (no quote tower)."""
    import re
    sender = original.get("from", "")
    subject = original.get("subject", "") or "(no subject)"
    if not subject.lower().startswith("re:"):
        subject = f"Re: {subject}"
    match = re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", sender)
    return {
        "to": match.group(0) if match else "",
        "subject": subject[:200],
        "body": reply_text.strip()[:10000],
        "in_reply_to": original.get("message_id", ""),
        "references": original.get("message_id", ""),
    }


class GmailReadInput(BaseModel):
    uid: int = Field(ge=1)


class GmailReplyInput(BaseModel):
    uid: int = Field(ge=1)
    body: str = Field(min_length=1, max_length=10000)


def _open():
    import imaplib
    address, password = creds()
    if not address or not password:
        return None, ToolResult(ok=False, error="Not configured: set GMAIL_ADDRESS + GMAIL_APP_PASSWORD.")
    try:
        box = imaplib.IMAP4_SSL("imap.gmail.com", 993)
        box.login(address, password)
        box.select("INBOX", readonly=True)
        return box, None
    except Exception:
        return None, ToolResult(ok=False, error="Gmail login failed: check address/app password.")


def _close(box) -> None:
    try:
        box.close()
    except Exception:
        pass
    try:
        box.logout()
    except Exception:
        pass


class GmailInboxTool(Tool):
    name = "gmail.inbox"
    description = ("List unread Gmail with sender/subject/date/uid plus a body "
                   "snippet. Auth is already handled — just call it, then use "
                   "gmail.read with a uid for the full text.")
    category = "connectors"
    risk_level = PermissionLevel.LOW
    input_model = GmailInboxInput
    timeout_seconds = 45.0

    def execute(self, validated: BaseModel) -> ToolResult:
        assert isinstance(validated, GmailInboxInput)
        try:
            import email
        except ImportError as exc:
            return ToolResult(ok=False, error=str(exc))
        box, err = _open()
        if err is not None:
            return err
        assert box is not None
        try:
            _, ids = box.search(None, "UNSEEN")
            seqs = (ids[0].split() if ids and ids[0] else [])[-validated.limit:][::-1]
            items = []
            for seq in seqs:
                _, udata = box.fetch(seq, "(UID BODY.PEEK[])")
                uid = 0
                raw = b""
                for part in udata:
                    if not isinstance(part, tuple):
                        continue
                    head = part[0] if len(part) > 0 else b""
                    if isinstance(head, bytes) and b"UID" in head:
                        try:
                            uid = int(head.split()[2].rstrip(b")"))
                        except (IndexError, ValueError):
                            pass
                    if len(part) > 1 and isinstance(part[1], bytes):
                        raw = part[1]
                if not raw:
                    continue
                msg = email.message_from_bytes(raw)
                full = body_text(msg, limit=2000)
                items.append({
                    "uid": uid,
                    "from": decode_header_value(str(msg.get("From", "")), 120),
                    "subject": decode_header_value(str(msg.get("Subject", "")), 200),
                    "date": str(msg.get("Date", ""))[:60],
                    "snippet": full[:300],
                })
            return ToolResult(ok=True, output={"unread": items, "count": len(items)})
        except Exception:
            return ToolResult(ok=False, error="Gmail inbox failed unexpectedly.")
        finally:
            _close(box)


class GmailReadTool(Tool):
    name = "gmail.read"
    description = ("Read one unread message's full body text by uid "
                   "(uids come from gmail.inbox). Auth is already handled.")
    category = "connectors"
    risk_level = PermissionLevel.LOW
    input_model = GmailReadInput
    timeout_seconds = 30.0

    def execute(self, validated: BaseModel) -> ToolResult:
        assert isinstance(validated, GmailReadInput)
        try:
            import email
        except ImportError as exc:
            return ToolResult(ok=False, error=str(exc))
        box, err = _open()
        if err is not None:
            return err
        assert box is not None
        try:
            _, data = box.uid("fetch", str(validated.uid), "(BODY.PEEK[])")
            raw = b""
            for part in data or []:
                if isinstance(part, tuple) and len(part) > 1 and isinstance(part[1], bytes):
                    raw = part[1]
                    break
            if not raw:
                return ToolResult(ok=False, error=f"No message with uid {validated.uid}.")
            msg = email.message_from_bytes(raw)
            return ToolResult(ok=True, output={
                "uid": validated.uid,
                "from": decode_header_value(str(msg.get("From", "")), 200),
                "to": decode_header_value(str(msg.get("To", "")), 200),
                "subject": decode_header_value(str(msg.get("Subject", "")), 200),
                "date": str(msg.get("Date", ""))[:80],
                "message_id": str(msg.get("Message-ID", ""))[:300],
                "body": body_text(msg, limit=8000),
            })
        except Exception:
            return ToolResult(ok=False, error="Gmail read failed unexpectedly.")
        finally:
            _close(box)


class GmailSendTool(Tool):
    name = "gmail.send"
    description = "Send an email via Gmail SMTP (needs approval). Auth is already handled."
    category = "connectors"
    risk_level = PermissionLevel.HIGH
    requires_confirmation = True
    input_model = GmailSendInput
    timeout_seconds = 30.0

    def execute(self, validated: BaseModel) -> ToolResult:
        assert isinstance(validated, GmailSendInput)
        address, password = creds()
        if not address or not password:
            return ToolResult(ok=False, error="Not configured: set GMAIL_ADDRESS + GMAIL_APP_PASSWORD.")
        try:
            import smtplib
            from email.message import EmailMessage
        except ImportError as exc:
            return ToolResult(ok=False, error=str(exc))
        msg = EmailMessage()
        msg["From"] = address
        msg["To"] = validated.to.strip()
        msg["Subject"] = validated.subject.strip()
        if validated.in_reply_to.strip():
            msg["In-Reply-To"] = validated.in_reply_to.strip()
            msg["References"] = (validated.references.strip()
                                 or validated.in_reply_to.strip())
        msg.set_content(validated.body)
        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=25) as smtp:
                smtp.login(address, password)
                smtp.send_message(msg)
            return ToolResult(ok=True, output={"sent": True, "to": validated.to.strip()})
        except Exception:
            return ToolResult(ok=False, error="Gmail send failed: check connection/credentials.")


class GmailReplyTool(Tool):
    name = "gmail.reply"
    description = ("Reply to a message by uid (uses its sender + subject with "
                   "Re: prefix). Needs approval. Auth is already handled.")
    category = "connectors"
    risk_level = PermissionLevel.HIGH
    requires_confirmation = True
    input_model = GmailReplyInput
    timeout_seconds = 45.0

    def execute(self, validated: BaseModel) -> ToolResult:
        assert isinstance(validated, GmailReplyInput)
        read = GmailReadTool().execute(GmailReadInput(uid=validated.uid))
        if not read.ok:
            return read
        composed = compose_reply(read.output, validated.body)
        if not composed["to"]:
            return ToolResult(ok=False, error="Could not find a reply address.")
        if not composed["body"]:
            return ToolResult(ok=False, error="Empty reply text.")
        return GmailSendTool().execute(GmailSendInput(**composed))
