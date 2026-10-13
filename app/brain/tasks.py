"""Lightweight multi-step task tracking (Phase 7).

The model drives adaptation — it sees every tool result. This module only
records what happened so the loop can (a) remind the model of progress on
later turns, (b) avoid repeating work, and (c) summarize honestly when the
budget runs out.

Simple questions never touch this: the tracker is created per `achat` call
and stays silent until the first tool actually executes.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StepRecord:
    """One executed (or reused) tool step."""

    tool: str
    ok: bool
    note: str = ""
    reused: bool = False


@dataclass
class TaskTracker:
    """Ordered ledger of tool steps for one user request."""

    goal: str = ""
    steps: list[StepRecord] = field(default_factory=list)

    def record(
        self, tool: str, ok: bool, note: str = "", reused: bool = False
    ) -> StepRecord:
        step = StepRecord(tool=tool, ok=ok, note=note, reused=reused)
        self.steps.append(step)
        return step

    @property
    def completed(self) -> int:
        return sum(1 for s in self.steps if s.ok and not s.reused)

    @property
    def failed(self) -> int:
        return sum(1 for s in self.steps if not s.ok)

    def progress_note(self) -> str:
        """Compact reminder injected before later model turns ('' if none)."""
        if not self.steps:
            return ""
        parts = []
        for i, step in enumerate(self.steps[-6:], start=max(1, len(self.steps) - 5)):
            status = "ok" if step.ok else "FAILED"
            tag = " (reused, not re-run)" if step.reused else ""
            parts.append(f"{i}. {step.tool} → {status}{tag}")
        return (
            "Task progress so far:\n" + "\n".join(parts)
            + "\nContinue toward the user's goal with the next step. "
            "Do not repeat a step that already succeeded; "
            "if a step failed, try a different approach or explain the blocker."
        )

    def summary(self) -> str:
        """Honest one-paragraph recap for budget-exhausted fallbacks."""
        total = len(self.steps)
        if not total:
            return "No tool steps completed."
        failed_names = [s.tool for s in self.steps if not s.ok]
        text = (
            f"I completed {self.completed} of {total} tool step(s)"
            f" ({self.failed} failed)."
        )
        if failed_names:
            text += f" Failed/blocked: {', '.join(failed_names)}."
        return text
