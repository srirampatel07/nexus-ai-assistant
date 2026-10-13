"""WhatsApp connector via Cloud API (honest gating, no faked sends).

Configure from the dashboard Connect panel (process-only env for testing):
WHATSAPP_TOKEN + WHATSAPP_PHONE_NUMBER_ID. Send requires approval.
Without credentials, `whatsapp.draft` formats a copy-paste message instead.
"""

from __future__ import annotations

import os

from pydantic import BaseModel, Field

from app.security.policy import PermissionLevel
from app.tools.base import Tool, ToolResult


class WhatsappSendInput(BaseModel):
    to: str = Field(min_length=5, max_length=32)
    text: str = Field(min_length=1, max_length=4000)


class WhatsappDraftInput(BaseModel):
    to: str = Field(default="", max_length=32)
    text: str = Field(min_length=1, max_length=4000)


class WhatsappSendTool(Tool):
    name = "whatsapp.send"
    description = "Send a WhatsApp text via Cloud API (needs approval + WHATSAPP_* env)."
    category = "connectors"
    risk_level = PermissionLevel.HIGH
    requires_confirmation = True
    input_model = WhatsappSendInput
    timeout_seconds = 30.0

    def execute(self, validated: BaseModel) -> ToolResult:
        assert isinstance(validated, WhatsappSendInput)
        token = os.environ.get("WHATSAPP_TOKEN", "")
        phone_id = os.environ.get("WHATSAPP_PHONE_NUMBER_ID", "")
        if not token or not phone_id:
            return ToolResult(ok=False, error="Not configured: set WHATSAPP_TOKEN + WHATSAPP_PHONE_NUMBER_ID.")
        try:
            import httpx
        except ImportError:
            return ToolResult(ok=False, error="httpx not installed.")
        try:
            r = httpx.post(
                f"https://graph.facebook.com/v21.0/{phone_id}/messages",
                headers={"Authorization": f"Bearer {token}"},
                json={"messaging_product": "whatsapp", "to": validated.to.strip(),
                      "type": "text", "text": {"body": validated.text}},
                timeout=20)
        except Exception as exc:
            return ToolResult(ok=False, error=f"WhatsApp request failed: {exc}")
        if r.status_code not in (200, 201):
            return ToolResult(ok=False, error=f"WhatsApp HTTP {r.status_code}.")
        return ToolResult(ok=True, output={"sent": True, "to": validated.to.strip()})


class WhatsappDraftTool(Tool):
    name = "whatsapp.draft"
    description = "Format a WhatsApp message to copy-paste (works with no credentials)."
    category = "connectors"
    risk_level = PermissionLevel.SAFE
    input_model = WhatsappDraftInput
    timeout_seconds = 15.0

    def execute(self, validated: BaseModel) -> ToolResult:
        assert isinstance(validated, WhatsappDraftInput)
        return ToolResult(ok=True, output={
            "to": validated.to.strip(), "text": validated.text.strip(),
            "hint": "Copy into WhatsApp manually, or configure WHATSAPP_* for direct send."})
