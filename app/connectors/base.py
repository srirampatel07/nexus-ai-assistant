"""Connector framework (Jarvis getjarvis.eu 30+ connectors parity).

Phase 1: explicit credential vault via environment only — never stored
in memory DB or logs. Each connector declares required env vars;
status reports configured/missing without leaking values.
Real OAuth flows are out of scope; PATs / API keys via env.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ConnectorSpec:
    key: str
    label: str
    env_vars: tuple[str, ...]
    description: str


CONNECTORS: tuple[ConnectorSpec, ...] = (
    ConnectorSpec("github", "GitHub", ("GITHUB_TOKEN",), "Repos, issues, PRs via API."),
    ConnectorSpec("gmail", "Gmail", ("GMAIL_ADDRESS", "GMAIL_APP_PASSWORD"), "IMAP/SMTP triage (app password)."),
    ConnectorSpec("whatsapp", "WhatsApp", ("WHATSAPP_TOKEN", "WHATSAPP_PHONE_NUMBER_ID"), "Cloud API send, or draft to copy."),
    ConnectorSpec("calendar", "Calendar", ("CALENDAR_ICS_PATH",), "Local ICS file reminders/events."),
    ConnectorSpec("slack", "Slack", ("SLACK_BOT_TOKEN",), "Thread summaries, posting."),
    ConnectorSpec("notion", "Notion", ("NOTION_TOKEN",), "Pages/databases search."),
    ConnectorSpec("linear", "Linear", ("LINEAR_API_KEY",), "Tickets from bugs."),
    ConnectorSpec("jira", "Jira", ("JIRA_URL", "JIRA_EMAIL", "JIRA_API_TOKEN"), "Tickets/boards/sprints."),
    ConnectorSpec("trello", "Trello", ("TRELLO_KEY", "TRELLO_TOKEN"), "Cards/boards."),
    ConnectorSpec("todoist", "Todoist", ("TODOIST_TOKEN",), "Tasks."),
    ConnectorSpec("asana", "Asana", ("ASANA_TOKEN",), "Tasks/projects."),
    ConnectorSpec("discord", "Discord", ("DISCORD_BOT_TOKEN",), "Channel summaries."),
    ConnectorSpec("obsidian", "Obsidian", ("OBSIDIAN_VAULT_PATH",), "Local vault search (allowed roots)."),
    ConnectorSpec("drive", "Drive", ("GOOGLE_DRIVE_TOKEN",), "Docs/sheets search (future)."),
    ConnectorSpec("spotify", "Spotify", ("SPOTIFY_TOKEN",), "Now playing / control (future)."),
    ConnectorSpec("weather", "Weather", (), "Keyless via web.search (no credential)."),
)


def connector_status() -> list[dict]:
    out: list[dict] = []
    for spec in CONNECTORS:
        missing = [v for v in spec.env_vars if not os.environ.get(v)]
        out.append({
            "key": spec.key,
            "label": spec.label,
            "configured": not missing,
            "missing_env": missing,
            "description": spec.description,
        })
    return out
