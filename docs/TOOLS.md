# NEXUS Tool Architecture (Phase 2)

NEXUS uses tools through a strict pipeline. No tool bypasses the executor:

```
User -> AI (tool definitions) -> ToolCall -> ToolExecutor
  -> validate -> permission check -> confirmation? -> execute (timeout)
  -> audit -> ToolResult -> AI -> final answer
```

## Modules

| Module | Role |
|---|---|
| `app/tools/base.py` | `Tool` ABC, `ToolResult`, dynamic `assess()` risk |
| `app/tools/registry.py` | `ToolRegistry`: register/unregister/get/list (unique names) |
| `app/tools/schemas.py` | Provider-neutral `LlmTool`/`LlmToolCall` + OpenAI wire mapping |
| `app/tools/permissions.py` | Facade over the security policy (no duplicated rules) |
| `app/tools/executor.py` | `ToolExecutor`: the only enforcement point |
| `app/tools/terminal.py` | `terminal.execute` + command classification |
| `app/tools/filesystem.py` | `filesystem.list/read/write/create_directory/delete` |
| `app/tools/applications.py` | `applications.launch` (allowlist only) |
| `app/tools/system.py` | `system.info/cpu/memory/platform` (read-only) |
| `app/security/policy.py` | `SAFE/LOW/MEDIUM/HIGH/BLOCKED` + `decide()` |
| `app/security/confirmation.py` | `ConfirmationRequest`, console/auto-deny callbacks |
| `app/security/audit.py` | JSON Lines audit (`logs/tools.jsonl`) + secret sanitizing |

## Permission model

- **SAFE** (system info, allowed dir listing): runs immediately.
- **LOW** (file read, allowlisted app launch): runs immediately.
- **MEDIUM** (file write/mkdir, sensitive commands): needs approval.
- **HIGH** (delete, destructive commands): needs approval.
- **BLOCKED** (format, credential theft, env dumping, ...): always denied.

Tools declare a static risk but may override `assess()` per call
(`terminal.execute` classifies each command as safe/confirm/blocked).

## Confirmation model

If the level or the tool requires it, the executor pauses and calls the
injected `confirm` callback with a `ConfirmationRequest` (tool, risk,
argument summary). The text CLI shows:

```
Tool request: terminal.execute
Description: ...
Risk: MEDIUM
Arguments: {...}
Allow? [y/N]
```

Non-interactive contexts default to **auto-deny**: the tool is skipped and
the denial is returned to the AI, which explains it to the user. Tools are
never executed silently.

## Allowed filesystem roots

Default: `C:\chanti\nexus` (configurable via `NEXUS_ALLOWED_ROOTS`).
All paths are resolved; `..` escapes and absolute paths outside roots are
rejected. Sensitive names (`.env`, private keys) are unreadable even inside
roots. Reads/writes are size-capped (`FILESYSTEM_MAX_READ/WRITE_BYTES`).

## Terminal security policy

- One PowerShell invocation per call, `shell=False`, hard timeout.
- Working directory must exist and sit inside allowed roots.
- Output truncated (`TERMINAL_MAX_OUTPUT_CHARS`) and secret-redacted.
- Environment is inherited but `Env:` dumping and `.env` reads are blocked.

**Limitation (honest):** regex classification is best-effort, not a sandbox.
Obfuscated commands can evade patterns. Real safety comes from confinement,
mandatory confirmation, audit logging, and redaction.

## Audit logging

One JSON event per execution in `logs/tools.jsonl`: timestamp, tool,
sanitized arguments, risk, permission decision, confirmation result,
status, duration, error. API keys, tokens, passwords, and Bearer credentials
are redacted before writing — never stored.

## How to add a new tool

1. Subclass `Tool` in `app/tools/<area>.py`: set `name`, `description`,
   `category`, `risk_level`, `requires_confirmation`, `timeout_seconds`,
   and a Pydantic `input_model`.
2. Implement `execute(validated) -> ToolResult` (sync; return failures,
   don't raise for expected errors). Override `assess()` for dynamic risk.
3. Register it in `build_default_registry()` in `app/tools/__init__.py`.
4. Add tests in `tests/` (registry, validation, policy, audit).
5. The AI receives its definition automatically — no assistant changes needed.
