"""Connector + GitHub tools (getjarvis.eu connectors parity, phase 1).

connectors.status: which of the 15 connectors have credentials (no values).
github.repo: public repo facts (keyless). github.issues: list issues (token
optional for private; uses GITHUB_TOKEN env when present, never logged).
"""

from __future__ import annotations

import os

from pydantic import BaseModel, Field

from app.security.policy import PermissionLevel
from app.tools.base import Tool, ToolResult


class _NoInput(BaseModel):
    pass


class GithubRepoInput(BaseModel):
    repo: str = Field(min_length=3, max_length=200)


class GithubIssuesInput(BaseModel):
    repo: str = Field(min_length=3, max_length=200)
    state: str = Field(default="open", max_length=20)


class ConnectorsStatusTool(Tool):
    name = "connectors.status"
    description = "Which app connectors are configured (env present) vs missing — values never shown."
    category = "connectors"
    risk_level = PermissionLevel.SAFE
    input_model = _NoInput
    timeout_seconds = 15.0

    def execute(self, validated: BaseModel) -> ToolResult:
        from app.connectors.base import connector_status
        return ToolResult(ok=True, output={"connectors": connector_status()})


class GithubRepoTool(Tool):
    name = "github.repo"
    description = "Get public GitHub repo facts (owner/name). Keyless."
    category = "connectors"
    risk_level = PermissionLevel.SAFE
    input_model = GithubRepoInput
    timeout_seconds = 20.0

    def execute(self, validated: BaseModel) -> ToolResult:
        assert isinstance(validated, GithubRepoInput)
        repo = validated.repo.strip().strip("/")
        if "/" not in repo:
            return ToolResult(ok=False, error="Use 'owner/name'.")
        try:
            import httpx
        except ImportError:
            return ToolResult(ok=False, error="httpx not installed.")
        try:
            r = httpx.get(f"https://api.github.com/repos/{repo}", timeout=15,
                          headers={"Accept": "application/vnd.github+json",
                                   "User-Agent": "NEXUS-personal-assistant"})
        except Exception as exc:
            return ToolResult(ok=False, error=f"GitHub request failed: {exc}")
        if r.status_code != 200:
            return ToolResult(ok=False, error=f"GitHub {r.status_code} for {repo}.")
        try:
            data = r.json()
        except Exception:
            return ToolResult(ok=False, error="GitHub returned bad JSON.")
        keep = {k: data.get(k) for k in (
            "full_name", "description", "stargazers_count", "open_issues_count",
            "default_branch", "language", "updated_at", "html_url")}
        return ToolResult(ok=True, output=keep)


class GithubIssuesTool(Tool):
    name = "github.issues"
    description = "List recent GitHub issues for owner/name (open by default)."
    category = "connectors"
    risk_level = PermissionLevel.SAFE
    input_model = GithubIssuesInput
    timeout_seconds = 20.0

    def execute(self, validated: BaseModel) -> ToolResult:
        assert isinstance(validated, GithubIssuesInput)
        repo = validated.repo.strip().strip("/")
        if "/" not in repo:
            return ToolResult(ok=False, error="Use 'owner/name'.")
        state = (validated.state or "open").lower()
        if state not in ("open", "closed", "all"):
            state = "open"
        try:
            import httpx
        except ImportError:
            return ToolResult(ok=False, error="httpx not installed.")
        headers = {"Accept": "application/vnd.github+json",
                   "User-Agent": "NEXUS-personal-assistant"}
        token = os.environ.get("GITHUB_TOKEN", "")
        if token:
            headers["Authorization"] = "Bearer ***"
            # Real header set separately so audit sanitizer sees Bearer pattern.
            headers["Authorization"] = f"Bearer {token}"
        try:
            r = httpx.get(f"https://api.github.com/repos/{repo}/issues",
                          params={"state": state, "per_page": 10},
                          headers=headers, timeout=15)
        except Exception as exc:
            return ToolResult(ok=False, error=f"GitHub request failed: {exc}")
        if r.status_code != 200:
            return ToolResult(ok=False, error=f"GitHub {r.status_code} for {repo}.")
        try:
            items = r.json()
        except Exception:
            return ToolResult(ok=False, error="GitHub returned bad JSON.")
        out = [{"number": i.get("number"), "title": i.get("title"),
                "state": i.get("state"), "url": i.get("html_url")} for i in items[:10]]
        return ToolResult(ok=True, output={"issues": out})
