# NEXUS Persistent Memory v1 (Phase 3)

Local SQLite-backed memory. The assistant persists conversation turns,
recalls relevant context into prompts, and honors explicit
"remember ..." requests — while refusing to ever store secrets.

## Architecture

```
assistant.py  ->  MemoryManager  ->  MemoryRepository  ->  SQLite (data/nexus.db)
                       |                     |
                  redaction.py          session-per-operation
                  (evaluate)            (never shared threads)
```

- `app/memory/categories.py` — controlled `MemoryCategory` enum + validator.
- `app/memory/models.py` — SQLAlchemy `MemoryEntry` (internal only).
- `app/memory/database.py` — engine factory, `init_db()` (create-if-missing,
  never destructive), `session_scope()` (commit/rollback/close).
- `app/memory/repository.py` — CRUD + LIKE search + soft delete.
- `app/memory/manager.py` — public API returning Pydantic `MemoryRecord`s.
- `app/memory/redaction.py` — secret detection (independent of audit.py).

## Database location

`data/nexus.db` by default (`MEMORY_DATABASE_PATH`). Created automatically
on first use when `MEMORY_ENABLED=true`. Excluded from git
(`data/*.db`). Never deleted or overwritten by the app.

## MemoryEntry fields

`id, conversation_id, category, content, importance (0-1), source
(chat|explicit|tool|system), created_at, updated_at, last_accessed_at,
is_deleted`. Indexes on conversation, category, deletion flag, creation time.

## Categories

`conversation, preference, project, fact, setting, tool_history`.
Unknown categories raise `ValueError`. `tool_history` additionally requires
`explicit_tool_history=True` — automatic tool-result persistence is refused,
so the DB never fills with raw tool output.

## MemoryManager API

- `remember(content, category, conversation_id, importance, source)` —
  validates, redacts/rejects secrets, returns `MemoryRecord`.
  Raises `ValueError` (empty/unknown category/tool_history gate) or
  `MemoryRejected` (secret material — nothing stored).
- `search(query, category, conversation_id, limit)` — substring match
  (empty query = importance/recency listing), deterministic order.
- `get(entry_id)` — one live entry; updates `last_accessed_at`.
- `get_context(conversation_id, query, limit)` — recent same-conversation
  turns (chronological) + global preference/fact/project/setting matches
  (importance → recency → id). Word-level fallback for sentence queries.
- `format_context(records)` — one bounded prompt section (`''` when empty).
- `get_recent(conversation_id, limit)` — newest-first conversation entries.
- `forget(entry_id, conversation_id?)` — soft delete (`is_deleted`).
- `clear_conversation(conversation_id)` — soft-delete all of one session.
- `count(...)`, `purge_expired()` (retention window enforcement).

## Redaction rules

Reject (store nothing): private-key blocks, raw tokens/keys (`gsk_*`,
`sk-*`, `xox*`, lone `Bearer` tokens), credential assignments
(`api_key=...`, `password: ...`, `.env`-style secret lines), bare
high-entropy tokens, blobs that are mostly secret lines. Placeholders
(`xxx`, `test`, `your-...`, `<key>`) never trigger. Secrets embedded in
longer prose are stored redacted (`***`); facts explicitly asked to be
remembered must be fully clean or the request is refused with an
explanation (the secret is never echoed back).

## Conversation IDs

Each `NexusAssistant` gets a `conversation_id` (generated 12-hex default,
explicit value preserved). Conversation-category entries stay scoped to
their session; preference/fact/project/setting entries are global, so facts
survive restarts. Shown in `/status`.

## Retrieval behavior (v1)

No embeddings, no vector DB: case-insensitive substring match over SQLite,
word-level fallback for natural questions, deterministic ordering
(importance desc, recency desc, id desc). Context injection order:
system instructions → memory section → recent history → current request →
tool loop. Bounded by `MEMORY_CONTEXT_LIMIT` entries and
`MEMORY_CONTEXT_MAX_CHARS` chars. Retrieval/persistence failures only log —
chat never breaks.

## CLI commands

- `/memory` — top memories across conversations (with conversation ids).
- `/memory search <text>` — global keyword search.
- `/forget <id>` — soft-delete one entry by id.
- `/status` — adds `memory_enabled`, `memory_database`,
  `conversation_id`, `memory_count` (current conversation).
- There is deliberately no bulk-delete command.

## Configuration (.env)

`MEMORY_ENABLED, MEMORY_DATABASE_PATH, MEMORY_CONTEXT_LIMIT,
MEMORY_CONTEXT_MAX_CHARS, MEMORY_RECENT_LIMIT, MEMORY_DEFAULT_CATEGORY,
MEMORY_RETENTION_DAYS` (0 = keep forever). See `.env.example`.

## Limitations of v1

Substring/keyword recall only (no semantic search); single local SQLite
file (no sync); no automatic summarization (long histories rely on
`MAX_HISTORY_MESSAGES` + retention); tool results are not auto-memorized.
