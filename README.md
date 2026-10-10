# NEXUS — Personal AI Operating Assistant

NEXUS is a modular, provider-independent personal AI operating assistant.
Phases 1–4 are complete (foundation, safe tool use, persistent memory,
hands-free voice). Phase 5.0 adds a local image-file vision interface
around the same assistant: point NEXUS at an image with `/vision` for a
Groq-hosted visual description. Phase 5.0 is implemented and awaiting
release. See `docs/VISION.md` for the vision guide and `docs/VOICE.md`
for the voice guide.

Personality: calm, intelligent, concise, helpful. NEXUS never pretends an
action succeeded when it did not.

## Phase 5.0 — Vision (implemented, awaiting release)

- Local image-file input (`/vision <image-path> [question]`) around the
  same `NexusAssistant` (memory + tools work unchanged)
- Pillow validation first: allowed-roots confinement, type/size caps,
  EXIF-orientation correction, transparency flattened onto white,
  EXIF/GPS stripped — then one upload to the vision model
- Groq-hosted vision (`VISION_MODEL`, default `qwen/qwen3.8-27b`) via the
  existing `AI_BASE_URL` + `AI_API_KEY`; the default text model
  (`openai/gpt-oss-120b`) is text-only and never receives images
- Ordinary chat and voice never upload images; offline metadata fallback
  when no vision backend is configured
- Full details in `docs/VISION.md`

## Phase 4 — Voice interface (complete)

- Microphone-driven conversation around the same `NexusAssistant`
  (memory + tools work unchanged; confirmations stay auto-deny in voice)
- Offline backends only: Vosk STT (local model) + Windows SAPI TTS —
  no cloud audio, no paid API, nothing recorded to disk
- `python run.py --voice` (stop phrase or Ctrl+C exits cleanly);
  `python run.py --voice --script "..."` for headless pipeline checks
- Full details in `docs/VOICE.md`

## Phase 3 — Persistent memory v1

- Local SQLite memory (`data/nexus.db`): conversation turns, facts,
  preferences, projects, settings — never secrets (refused or redacted)
- `MemoryManager` API: `remember/search/get/get_context/get_recent/forget/clear_conversation`
- Deterministic keyword recall (no embeddings yet); per-conversation scoping
  with global facts that survive restarts
- Each assistant run gets a `conversation_id` (shown in `/status`)
- CLI: `/memory`, `/memory search <text>`, `/forget <id>`
- Full details in `docs/MEMORY.md`

Phase 2 tool system (registry, permissions, confirmations, audit) is
unchanged underneath.

## Phase 2 — Safe tool use

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

IoT, computer automation, API, and dashboard arrive in later phases.

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

Text commands: `/help` `/tools` `/memory` `/memory search <text>` `/forget <id>` `/clear` `/status` `/vision <image-path> [question]` `/exit`.

## Memory settings

| Variable | Default | Purpose |
|---|---|---|
| `MEMORY_ENABLED` | `true` | persistent memory on/off |
| `MEMORY_DATABASE_PATH` | `data/nexus.db` | local SQLite file |
| `MEMORY_CONTEXT_LIMIT` | `10` | max memory entries per prompt |
| `MEMORY_CONTEXT_MAX_CHARS` | `4000` | max memory prompt chars |
| `MEMORY_RECENT_LIMIT` | `10` | recent turns recalled |
| `MEMORY_DEFAULT_CATEGORY` | `conversation` | default remember category |
| `MEMORY_RETENTION_DAYS` | `0` | expiry window (0 = keep forever) |

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

## Project structure (Phase 5.0)

```
NEXUS/
  app/__init__.py  app/main.py
  app/core/   config.py logger.py events.py exceptions.py
  app/brain/  assistant.py planner.py router.py provider.py prompts.py
  app/tools/  base.py registry.py schemas.py permissions.py executor.py
              terminal.py filesystem.py applications.py system.py
  app/security/  policy.py confirmation.py audit.py
  app/memory/  database.py models.py repository.py manager.py
               redaction.py categories.py
  app/voice/  audio.py stt.py tts.py session.py factories.py errors.py
  app/vision/  loader.py describer.py factories.py errors.py
  tests/      conftest + test_config/provider/assistant/core/prompts
              + test_tool_registry/audit/filesystem/terminal/apps_system
              + test_executor/tool_calls/assistant_tools/tool_wire
              + test_memory/memory_integration
              + test_voice_audio/backends/session/cli
              + test_vision_loader/describer/assistant/security
  docs/TOOLS.md docs/MEMORY.md docs/VOICE.md docs/VISION.md
  data/ logs/ scripts/ frontend/dashboard/  (placeholders for later phases)
  run.py  requirements.txt  .env.example  README.md
```

## Roadmap

- Phase 2: tool registry, computer/filesystem/terminal tools, permissions, audit
- Phase 3: SQLite memory + context retrieval
- Phase 4: voice interface (offline STT/TTS, hands-free)
- Phase 5.0: vision (local image-file input, Groq-hosted description) — implemented, awaiting release
- Later: screenshots/camera + safe UI actions, IoT (MQTT/ESP32), API, dashboard, autonomous multi-step tasks
