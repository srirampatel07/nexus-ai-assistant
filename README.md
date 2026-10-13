# NEXUS — Personal AI Operating Assistant

NEXUS is a Python-based assistant using NVIDIA cloud APIs. Text chat uses
`nvidia/nemotron-3.5-lightning-30b-a3b`; cloud vision uses the separate
vision-capable `meta/llama-3.2-11b-vision-instruct` (the text model does not
accept images). Both go through `https://integrate.api.nvidia.com/v1` with a
Bearer API key. There is no local inference in normal operation.

Phases 1–6 are complete (foundation, safe tool use, persistent memory,
hands-free voice, cloud vision, web search + reader browser). Phase 7 adds
bounded multi-step tool orchestration: dedup, truncation, timeouts, progress
notes, and a tool-free synthesis turn with honest budget-exhausted fallbacks.
See `docs/TOOLS.md` (tools + Phase 7 loop), `docs/WEB.md` (web), `docs/VISION.md`
(vision), `docs/VOICE.md` (voice), `docs/MEMORY.md` (memory), and
`docs/REPOSITORY_AUDIT.md` (pre–Phase 7 audit).

Personality: calm, intelligent, concise, helpful. NEXUS never pretends an
action succeeded when it did not.

## Phase 7 — Agent orchestration (in this tree, awaiting commit)

- Bounded loop per reply: `MAX_TOOL_ITERATIONS=3` model turns, `TOOL_LOOP_TIMEOUT_SECONDS=300` wall-clock (first turn always runs).
- Every returned call in a turn runs in order; identical repeats reuse the cached result (no duplicate side effects).
- One oversized tool result is cut at `TOOL_RESULT_MAX_CHARS=20000` with a truncation marker.
- A per-request tracker feeds compact progress notes to later turns; budget exhaustion ends with a tool-free synthesis turn or an honest summary naming completed steps and blockers.
- Empty model turns never produce blank replies. Details in `docs/TOOLS.md`.

## Phase 6.0 — Web + browser

- Web search (`web.search`, keyless) and page retrieval (`web.fetch`) through the standard tool pipeline.
- Reader browser sessions (`browser.navigate/read/links/click/back/close`); `browser.click` needs approval.
- Visible Brave launcher (`brave.open`, LOW, no approval): public http(s) URLs or current-search references (`result 2`, `second`, `first`, `last`).
- SSRF protection on every request and redirect hop; size/type caps; readable-text extraction with source URLs.
- Web content is untrusted evidence: cited, never obeyed. Details in `docs/WEB.md`.

## Phase 5.0 — Vision (NVIDIA cloud)

- Explicit image-file input (`/vision <image-path> [question]`) around the same `NexusAssistant`.
- Pillow validation first (allowed-roots confinement, type/size caps, EXIF-orientation correction, transparency flattened, EXIF/GPS stripped), then one upload to the NVIDIA-hosted vision model.
- The chat model never receives images. Ordinary chat and voice never upload images.
- The retired ID `nvidia/nemotron-nano-12b-v2-vl` (retired 2026-08-26) is ignored by the resolver even if left in `.env`; health tells you to remove it. Details in `docs/VISION.md`.

## Phase 4 — Voice (Phase 8: NVIDIA cloud speech)

- Live microphone speech-to-text works via verified NVIDIA cloud Parakeet
  ASR (`nvidia/parakeet-ctc-1_1b-asr`, English, NVCF HTTPS —
  verified live Oct 2026). `run.py --voice` captures the mic only after
  explicit startup, transcribes with NVIDIA, and answers through the
  existing Nemotron assistant (tools + memory). No local AI models.
- Headless pipeline checks work: `run.py --voice --script "..."` (canned
  transcripts, no mic, no network). Spoken replies default to the local
  Windows OS synthesizer (SAPI — an OS utility, not NVIDIA speech
  generation); `VOICE_TTS_BACKEND=nvidia` selects verified NVIDIA cloud
  Magpie TTS instead. Text mode is unaffected. Details in `docs/VOICE.md`.

## Phase 3 — Persistent memory v1

- Local SQLite memory (`data/nexus.db`): turns, facts, preferences, projects, settings — never secrets.
- Deterministic keyword recall (no embeddings); per-conversation scoping with global facts.
- CLI: `/memory`, `/memory search <text>`, `/forget <id>`. Details in `docs/MEMORY.md`.

## Phase 2 — Safe tool use

- 20 tools when web is enabled (12 always: system/filesystem/terminal/applications/Brave + 8 web/browser).
- Permissions SAFE / LOW / MEDIUM / HIGH / BLOCKED; MEDIUM+ pauses for `Allow? [y/N]`, auto-deny when non-interactive.
- Terminal: PowerShell, `shell=False`, timeout, output caps, command policy. Filesystem confined to `NEXUS_ALLOWED_ROOTS`. App allowlist (`notepad, explorer, powershell, code`).
- Audit: every execution in `logs/tools.jsonl` with secrets redacted. Unexpected tool exceptions become audited failures, never chat crashes.
- CLI: `/tools` lists tools with risk levels.

## Quickstart (Windows)

```powershell
# From C:\chanti\nexus:
uv venv
.venv\Scripts\activate
uv pip install -r requirements.txt   # or: pip install -r requirements.txt

# Configure privately (never commit .env):
copy .env.example .env
# Edit .env: set NVIDIA_API_KEY to your key, and:
#   AI_PROVIDER=nvidia
#   NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
#   NVIDIA_MODEL=nvidia/nemotron-3.5-lightning-30b-a3b
#   NVIDIA_VISION_MODEL=meta/llama-3.2-11b-vision-instruct
# Or double-click Start-NEXUS.bat and use [2] Health Check first.

# Run
.venv\Scripts\python.exe run.py --health
.venv\Scripts\python.exe run.py --text
.venv\Scripts\python.exe run.py --text --debug
```

Or double-click **`Start-NEXUS.bat`** → `[1] Start NEXUS (NVIDIA cloud)`. No venv activation, no servers, no ports.

Text commands: `/help` `/tools` `/memory` `/memory search <text>` `/forget <id>` `/clear` `/status` `/vision <image-path> [question]` `/exit`.

## Configuration (source of truth: `app/core/config.py`)

| Variable | Default | Purpose |
|---|---|---|
| `AI_PROVIDER` | `echo` | `echo` (offline) / `openai` / `openai-compatible` / `nvidia` |
| `AI_MODEL` / `AI_BASE_URL` / `AI_API_KEY` | `nexus-default` / _(empty)_ / _(empty)_ | generic OpenAI-compatible override; hosted NVIDIA IDs here take precedence over `NVIDIA_*` |
| `NVIDIA_API_KEY` | _(empty)_ | preferred cloud key (`AI_API_KEY` accepted as fallback); never logged |
| `NVIDIA_BASE_URL` | `https://integrate.api.nvidia.com/v1` | cloud endpoint |
| `NVIDIA_MODEL` | `nvidia/nemotron-3.5-lightning-30b-a3b` | cloud text model |
| `NVIDIA_VISION_MODEL` | `meta/llama-3.2-11b-vision-instruct` | cloud vision model |
| `AI_TIMEOUT_SECONDS` | `30` | provider HTTP timeout |
| `MAX_HISTORY_MESSAGES` | `20` | bounded context window |
| `MAX_TOOL_ITERATIONS` | `3` | agentic loop turn cap |
| `TOOL_LOOP_TIMEOUT_SECONDS` | `300` | wall-clock budget per reply |
| `TOOL_RESULT_MAX_CHARS` | `20000` | one tool result fed back to the model |
| `LOG_LEVEL` | `INFO` | `DEBUG/INFO/WARNING/ERROR` |

Loopback `AI_BASE_URL` values and GGUF-path `AI_MODEL` values left over from legacy local configs are ignored in NVIDIA mode (the official defaults win). There are no other AI providers configured: local operation examples were deleted from `.env.example`, and `VISION_PROVIDER=local` raises an explicit error.

Memory, voice, vision, and web settings live in `.env.example` with the same names the code reads (`MEMORY_*`, `VOICE_*`, `VISION_*`, `WEB_*`). Voice: live STT uses verified NVIDIA cloud Parakeet (`VOICE_STT_BACKEND=nvidia`, needs `NVIDIA_API_KEY`); spoken output uses the local Windows OS synthesizer (`sapi`), NVIDIA cloud Magpie (`nvidia`), or silent mode (`null`). Search default is the keyless `https://html.duckduckgo.com/html/`; the legacy Instant Answer endpoint still parses when configured.

## Provider availability, quotas, and errors

No unlimited or free usage is promised. The NVIDIA endpoint may rate-limit (429), reject keys (401/403), report quota/payment failures (402/429), or fail upstream (5xx); `run.py --health` categorizes these (`ok | missing_credentials | auth_error | rate_limited | quota_error | bad_request | provider_error | timeout | network_error`) without printing secrets. Vision failures, tool errors, and memory errors are reported honestly and never leak credentials. The retired vision ID above is never sent.

## Tests

```powershell
.venv\Scripts\python.exe -m pytest -q
```

Tests mock all external services and need no API credentials (621 passed, 1 skipped at audit time). A small live `run.py --health` check is safe when a key is configured; it consumes no chat quota and uploads no images. Avoid repeated live vision/chat calls.

## Project structure (Phase 8 Jarvis parity)

```
NEXUS/
  app/__init__.py  app/main.py
  app/core/   config.py logger.py events.py exceptions.py
  app/brain/  assistant.py planner.py router.py provider.py prompts.py tasks.py budget.py
  app/tools/  base.py registry.py schemas.py permissions.py executor.py
              terminal.py filesystem.py applications.py system.py
              web.py browser.py brave.py
              screen.py git.py docker.py reminders.py routines.py connectors.py imagegen.py
  app/screen/  capture.py overlay.py
  app/personas/  catalog.py
  app/routines/  store.py
  app/connectors/  base.py
  app/server/  briefing.py
  app/security/  policy.py confirmation.py audit.py
  app/memory/  database.py models.py repository.py manager.py
               redaction.py categories.py
  app/voice/  audio.py stt.py tts.py session.py factories.py errors.py
  app/vision/  loader.py describer.py factories.py errors.py
  app/web/  ssrf.py extract.py client.py search_session.py errors.py
  app/browser/  session.py errors.py
  app/launcher/  menu.py   (cloud-only menu; local helpers removed)
  tests/      conftest + config/provider/assistant/core/prompts
              + tool registry/audit/filesystem/terminal/apps_system
              + executor/tool_calls/assistant_tools/tool_wire/names/loop
              + nvidia provider/vision + nvidia-only + phase7 orchestration
              + memory/integration + voice + vision + web/browser/brave
              + gemini compat + provider transport + jarvis_parity
  docs/  REPOSITORY_AUDIT.md TOOLS.md MEMORY.md VOICE.md VISION.md WEB.md LOCAL.md JARVIS.md
  data/ logs/ scripts/ frontend/dashboard/  (runtime data + phase placeholders)
  run.py  requirements.txt  .env.example  README.md
  Start-NEXUS.bat
```

## Roadmap

- Phase 2: tool registry, permissions, audit
- Phase 3: SQLite memory + context retrieval
- Phase 4: voice interface (local STT/TTS, hands-free)
- Phase 5.0: vision (explicit image-file input, NVIDIA-hosted description)
- Phase 6.0: web search, page retrieval, reader browser sessions
- Phase 7: bounded multi-step tool orchestration (this tree — dedup, truncation, timeout, progress notes, synthesis turn)
- Phase 8: Jarvis parity foundation (this tree — screen.capture/point, personas/tones, git/docker, reminders/routines, connectors.status+github, image.generate, smart filter + budget guards, --briefing Command Center). See `docs/JARVIS.md`.
- Later: full OAuth connectors, background daemon + nudges, HTTP API + rich dashboard, wake-word + 27-lang voice, mobile remote, IoT (MQTT/ESP32)

Legacy local operation (llama.cpp servers, GGUF files, `switch-model.ps1`,
local vision) was removed; `docs/LOCAL.md` records the removal and the
migration. The cloud launcher never starts model servers.
