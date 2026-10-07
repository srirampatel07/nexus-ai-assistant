# NEXUS — Personal AI Operating Assistant

NEXUS is a modular, provider-independent personal AI operating assistant.
Phase 2 (this release) adds a **safe, modular tool system**: the assistant
can inspect the system, work with files in allowed folders, run vetted
commands, and launch approved apps — every call validated, permission-checked,
confirmed when risky, and audit-logged. See `docs/TOOLS.md` for the full
tool architecture.

Personality: calm, intelligent, concise, helpful. NEXUS never pretends an
action succeeded when it did not.

## Phase 2 — Safe tool use (current)

- Tool registry (`app/tools`): 11 tools across system/filesystem/terminal/applications
- Permission model: SAFE / LOW / MEDIUM / HIGH / BLOCKED (`app/security/policy.py`)
- Confirmation: MEDIUM+ tools pause for `Allow? [y/N]`; auto-deny when non-interactive
- Terminal: PowerShell, `shell=False`, timeout, output caps, blocked/confirm command policy
- Filesystem confined to `NEXUS_ALLOWED_ROOTS` (default `C:\chanti\nexus`); `.env`/keys protected
- App launcher allowlist (`notepad, explorer, powershell, code`)
- Audit: every execution in `logs/tools.jsonl` with secrets redacted
- AI tool loop: OpenAI-compatible `tools`/`tool_calls` with safe fallback (never faked)
- CLI: `/tools` lists registered tools with risk levels

Phase 1 foundation (config, logging, provider abstraction, bounded context,
text CLI) is unchanged underneath.

Advanced voice, vision, IoT, computer automation, memory DB, API, and
dashboard arrive in Phases 2–9. `python run.py --voice` explains this and
exits cleanly instead of crashing.

## Quickstart (Windows)

```powershell
# 1. Install Python 3.11+ (example with uv, no admin needed)
# 2. From C:\chanti\nexus:
uv venv
.venv\Scripts\activate
uv pip install -r requirements.txt   # or: pip install -r requirements.txt

# 3. Configure (offline defaults work out of the box)
copy .env.example .env

# 4. Run
python run.py --health
python run.py --text
python run.py --text --debug   # visible provider/history info
```

Text commands: `/help` `/tools` `/clear` `/status` `/exit`.

## New tool settings

| Variable | Default | Purpose |
|---|---|---|
| `NEXUS_ALLOWED_ROOTS` | `C:\chanti\nexus` | filesystem + terminal confinement |
| `NEXUS_ALLOWED_APPS` | `notepad,explorer,powershell,code` | launch allowlist |
| `TERMINAL_DEFAULT_TIMEOUT_SECONDS` | `30` | per-command timeout |
| `TERMINAL_MAX_OUTPUT_CHARS` | `65536` | stdout/stderr cap |
| `FILESYSTEM_MAX_READ_BYTES` / `FILESYSTEM_MAX_WRITE_BYTES` | `1048576` | file size caps |
| `TOOLS_AUDIT_FILE` | `logs/tools.jsonl` | audit log path |
| `MAX_TOOL_ITERATIONS` | `3` | agentic loop cap |

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `AI_PROVIDER` | `echo` | `echo` (offline) or `openai`/`openai-compatible` |
| `AI_MODEL` | `nexus-default` | model name sent to provider |
| `AI_API_KEY` | _(empty)_ | never logged; required for `openai` |
| `AI_BASE_URL` | _(empty)_ | e.g. `https://api.openai.com/v1` |
| `AI_TIMEOUT_SECONDS` | `30` | provider HTTP timeout |
| `MAX_HISTORY_MESSAGES` | `20` | bounded context window |
| `LOG_LEVEL` | `INFO` | `DEBUG/INFO/WARNING/ERROR` |

The provider stays generic: any OpenAI-compatible endpoint works via
`AI_BASE_URL` + `AI_MODEL` + `AI_API_KEY` (nothing is hard-coded).
Currently working configuration (Groq):

```env
AI_PROVIDER=openai-compatible
AI_MODEL=openai/gpt-oss-120b
AI_API_KEY=your-groq-api-key
AI_BASE_URL=https://api.groq.com/openai/v1
```

## Tests

```powershell
python -m pytest -q
```

Tests mock all external services and need no API credentials.

## Project structure (Phase 1)

```
NEXUS/
  app/__init__.py  app/main.py
  app/core/   config.py logger.py events.py exceptions.py
  app/brain/  assistant.py planner.py router.py provider.py prompts.py
  app/tools/  base.py registry.py schemas.py permissions.py executor.py
              terminal.py filesystem.py applications.py system.py
  app/security/  policy.py confirmation.py audit.py
  tests/      conftest + test_config/provider/assistant/core/prompts
              + test_tool_registry/audit/filesystem/terminal/apps_system
              + test_executor/tool_calls/assistant_tools
  docs/TOOLS.md
  data/ logs/ scripts/ frontend/dashboard/  (placeholders for later phases)
  run.py  requirements.txt  .env.example  README.md
```

## Roadmap

- Phase 2: tool registry, computer/filesystem/terminal tools, permissions, audit
- Phase 3: SQLite memory + context retrieval
- Phase 4: FastAPI + WebSocket
- Phase 5: voice (STT/TTS/wake-word)
- Phase 6: vision (screenshots + safe UI actions)
- Phase 7: IoT (MQTT/ESP32)
- Phase 8: dashboard
- Phase 9: autonomous multi-step tasks
