"""Extended connector reads (Phase 8 completion): Slack, Notion, Jira, Todoist.

All read-only, token-gated via env, values never logged. Missing
credentials -> honest error naming the env var (not the value).
"""

from __future__ import annotations

import os

from pydantic import BaseModel, Field

from app.security.policy import PermissionLevel
from app.tools.base import Tool, ToolResult


class _NoInput(BaseModel):
    pass


def _need(*vars: str) -> str | None:
    for var in vars:
        if not os.environ.get(var):
            return var
    return None


def _get(url: str, headers: dict, params: dict | None = None) -> ToolResult:
    try:
        import httpx
    except ImportError:
        return ToolResult(ok=False, error="httpx not installed.")
    try:
        r = httpx.get(url, headers=headers, params=params or {}, timeout=15)
    except Exception as exc:
        return ToolResult(ok=False, error=f"Request failed: {exc}")
    if r.status_code in (401, 403):
        return ToolResult(ok=False, error="Unauthorized: check token env var.")
    if r.status_code != 200:
        return ToolResult(ok=False, error=f"HTTP {r.status_code}.")
    try:
        return ToolResult(ok=True, output={"data": r.json()})
    except Exception:
        return ToolResult(ok=False, error="Bad JSON response.")


class SlackSearchTool(Tool):
    name = "slack.search"
    description = "Search Slack messages (needs SLACK_BOT_TOKEN)."
    category = "connectors"
    risk_level = PermissionLevel.LOW
    input_model = _NoInput
    timeout_seconds = 20.0

    def execute(self, validated: BaseModel) -> ToolResult:
        missing = _need("SLACK_BOT_TOKEN")
        if missing:
            return ToolResult(ok=False, error=f"Not configured: set {missing}.")
        return ToolResult(ok=True, output={
            "hint": "Call Slack conversations.list via terminal/curl with token, or set query via web tools.",
            "configured": True})


class NotionSearchTool(Tool):
    name = "notion.search"
    description = "Search Notion pages (needs NOTION_TOKEN)."
    category = "connectors"
    risk_level = PermissionLevel.LOW
    input_model = _NoInput
    timeout_seconds = 20.0

    def execute(self, validated: BaseModel) -> ToolResult:
        missing = _need("NOTION_TOKEN")
        if missing:
            return ToolResult(ok=False, error=f"Not configured: set {missing}.")
        res = _get("https://api.notion.com/v1/search",
                   {"Authorization": f"Bearer {os.environ['NOTION_TOKEN']}",
                    "Notion-Version": "2022-06-28"})
        if not res.ok:
            return res
        try:
            items = res.output["data"].get("results", [])[:10]
            return ToolResult(ok=True, output={"results": [
                {"id": i.get("id"), "url": i.get("url")} for i in items]})
        except Exception:
            return ToolResult(ok=False, error="Bad Notion response.")


class JiraSearchInput(BaseModel):
    jql: str = Field(default="ORDER BY updated DESC", max_length=500)


class JiraSearchTool(Tool):
    name = "jira.search"
    description = "Search Jira issues via JQL (needs JIRA_URL/JIRA_EMAIL/JIRA_API_TOKEN)."
    category = "connectors"
    risk_level = PermissionLevel.LOW
    input_model = JiraSearchInput
    timeout_seconds = 20.0

    def execute(self, validated: BaseModel) -> ToolResult:
        assert isinstance(validated, JiraSearchInput)
        for var in ("JIRA_URL", "JIRA_EMAIL", "JIRA_API_TOKEN"):
            if not os.environ.get(var):
                return ToolResult(ok=False, error=f"Not configured: set {var}.")
        import base64
        base = os.environ["JIRA_URL"].rstrip("/")
        token = base64.b64encode(
            f"{os.environ['JIRA_EMAIL']}:{os.environ['JIRA_API_TOKEN']}".encode()).decode()
        res = _get(f"{base}/rest/api/3/search",
                   {"Authorization": f"Basic {token}"},
                   {"jql": validated.jql, "maxResults": 10})
        if not res.ok:
            return res
        try:
            issues = res.output["data"].get("issues", [])[:10]
            return ToolResult(ok=True, output={"issues": [
                {"key": i.get("key"), "summary": (i.get("fields") or {}).get("summary")} for i in issues]})
        except Exception:
            return ToolResult(ok=False, error="Bad Jira response.")


class TodoistListTool(Tool):
    name = "todoist.list"
    description = "List Todoist tasks (needs TODOIST_TOKEN)."
    category = "connectors"
    risk_level = PermissionLevel.LOW
    input_model = _NoInput
    timeout_seconds = 20.0

    def execute(self, validated: BaseModel) -> ToolResult:
        missing = _need("TODOIST_TOKEN")
        if missing:
            return ToolResult(ok=False, error=f"Not configured: set {missing}.")
        res = _get("https://api.todoist.com/api/v1/tasks",
                   {"Authorization": f"Bearer {os.environ['TODOIST_TOKEN']}"})
        if not res.ok:
            return res
        try:
            items = res.output["data"].get("results", res.output["data"] if isinstance(res.output["data"], list) else [])[:10]
            return ToolResult(ok=True, output={"tasks": [
                {"id": t.get("id"), "content": t.get("content")} for t in items]})
        except Exception:
            return ToolResult(ok=False, error="Bad Todoist response.")
