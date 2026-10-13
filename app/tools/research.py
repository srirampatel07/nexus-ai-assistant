"""Deep-research helper (Jarvis ceo parity): 2-search evidence pack.

No AI synthesis here — the tool gathers cited snippets; the model writes
the report with source URLs (Phase 7 loop may call it + web.fetch).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.security.policy import PermissionLevel
from app.tools.base import Tool, ToolResult
from app.web.client import WebContext


class ResearchInput(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class ResearchReportTool(Tool):
    name = "research.report"
    description = "Gather top web evidence for a big question (2 searches). Model writes the cited report."
    category = "web"
    risk_level = PermissionLevel.LOW
    input_model = ResearchInput
    timeout_seconds = 60.0

    def __init__(self, ctx: WebContext | None = None) -> None:
        self.ctx = ctx

    def execute(self, validated: BaseModel) -> ToolResult:
        assert isinstance(validated, ResearchInput)
        from app.web.client import search_ddg
        from app.web.errors import WebError

        ctx = self.ctx or WebContext()
        q = validated.question.strip()
        queries = [q, f"{q} latest details"]
        evidence: list[dict] = []
        for query in queries:
            try:
                found = search_ddg(ctx, query)
            except WebError as exc:
                evidence.append({"query": query, "error": str(exc)})
                continue
            for r in found.results[:5]:
                evidence.append({"query": query, "title": r.title,
                                 "url": r.url, "snippet": r.snippet})
        if not evidence:
            return ToolResult(ok=False, error="No evidence found.")
        return ToolResult(ok=True, output={
            "question": q, "evidence": evidence[:10],
            "hint": "Write the report now with cited source URLs."})
