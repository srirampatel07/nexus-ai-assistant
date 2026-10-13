"""Session budget guards (Jarvis institute cost-guard parity).

Token estimate: chars/4 heuristic (documented, no tokenizer dep).
Enforced per-reply tool loop + session caps from Settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def estimate_tokens(text: str) -> int:
    return max(0, (len(text or "") + 3) // 4)


@dataclass
class BudgetGuard:
    max_session_tokens: int = 100000
    max_reply_tokens: int = 20000
    used_session_tokens: int = 0
    _reply_tokens: int = field(default=0, repr=False)

    def start_reply(self) -> None:
        self._reply_tokens = 0

    def charge(self, text: str) -> int:
        n = estimate_tokens(text)
        self._reply_tokens += n
        self.used_session_tokens += n
        return n

    def reply_over(self) -> bool:
        return self._reply_tokens > max(0, self.max_reply_tokens)

    def session_over(self) -> bool:
        return self.used_session_tokens > max(0, self.max_session_tokens)


RELEVANCE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "system.": ("system", "cpu", "memory", "platform", "version", "computer"),
    "filesystem.": ("file", "folder", "directory", "read", "write", "list", "path"),
    "terminal.": ("run", "command", "powershell", "script", "exec", "terminal", "install"),
    "applications.": ("open", "launch", "app", "notepad", "browser", "start"),
    "web.": ("search", "web", "news", "weather", "lookup", "find online"),
    "browser.": ("page", "browse", "click", "link", "navigate", "site"),
    "brave.": ("open browser", "brave", "show me", "visible"),
    "screen.": ("screen", "screenshot", "show", "point", "circle", "display", "monitor"),
    "git.": ("git", "commit", "repo", "branch", "pr", "diff", "pull"),
    "docker.": ("docker", "container", "image", "compose"),
    "routines.": ("routine", "schedule", "recurring", "daily", "reminder routine", "cron"),
    "reminders.": ("remind", "reminder", "todo", "task", "calendar", "event"),
    "connectors.": ("connect", "connector", "status", "configured"),
    "github.": ("github", "repo", "issue", "pr", "pull"),
    "gmail.": ("gmail", "mail", "inbox", "unread", "email", "send", "read", "reply", "body"),
    "whatsapp.": ("whatsapp", "message", "chat", "send", "draft"),
    "slack.": ("slack", "message", "thread", "channel"),
    "notion.": ("notion", "page", "notes"),
    "jira.": ("jira", "ticket", "sprint", "board", "issue"),
    "todoist.": ("todoist", "todo", "task"),
    "linear.": ("linear", "ticket", "issue"),
    "research.": ("research", "report", "deep", "investigate", "sources"),
    "image.": ("image", "picture", "generate", "draw", "wallpaper"),
}


_ALWAYS_KEEP = ("system.info", "connectors.status")


def _query_words(query: str) -> list[str]:
    words: list[str] = []
    for token in (query or "").lower().split():
        word = "".join(ch for ch in token if ch.isalnum())
        if len(word) >= 3 and word not in words:
            words.append(word)
    return words


def _tool_text(name: str, description: str = "") -> str:
    return f"{name.replace('.', ' ').replace('_', ' ')} {description}".lower()


def select_relevant_tools(
    tool_names: list[str],
    query: str,
    *,
    max_tools: int = 12,
    descriptions: dict[str, str] | None = None,
) -> list[str]:
    """Keyword pre-filter to save context (~institute smart selection).

    Scores each tool by word overlap between the query and the tool's
    name + description (plus prefix-keyword bonuses). Always keeps
    system.info + connectors.status; returns all when the query is empty
    or nothing matches. Deterministic, no AI call.
    """
    q = (query or "").lower()
    if not q.strip():
        return list(tool_names)
    words = _query_words(query)
    descriptions = descriptions or {}
    scored: list[tuple[int, int, str]] = []
    for index, name in enumerate(tool_names):
        text = _tool_text(name, descriptions.get(name, ""))
        score = sum(1 for w in words if w in text)
        prefix = name.split(".")[0] + "."
        for key in RELEVANCE_KEYWORDS.get(prefix, ()):
            if key in q:
                score += 1
        scored.append((score, index, name))
    hits = [(s, i, n) for s, i, n in scored if s > 0]
    if not hits:
        return list(tool_names)
    # Top matches by score, bounded — then restore registry order.
    hits.sort(key=lambda t: (-t[0], t[1]))
    keep = {n for _, _, n in hits[: max(4, max_tools)]}
    for helper in _ALWAYS_KEEP:
        if helper in tool_names:
            keep.add(helper)
    return [n for n in tool_names if n in keep]
