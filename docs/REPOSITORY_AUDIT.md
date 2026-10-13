# NEXUS Repository Audit (pre-Phase 7 commit)

Date: 2026-10-13. Branch: `main` (ahead of `origin/main`, uncommitted Phase 7 + NVIDIA cloud work).
Baseline tests: **621 passed, 1 skipped** (`.\.venv\Scripts\python.exe -m pytest -q`, ~43s).
No `AGENTS.md` exists. `.env` is untracked (good); `.env.backup-local-model` is git-ignored (good).

## 1. Actual current architecture and entry points

Python assistant, NVIDIA cloud-only for AI inference. No local inference in the normal path.

- Entry points:
  - `run.py` — CLI: `--text` (tools+vision+web), `--voice [--script]`, `--health [--debug]`, `--version`. Text commands: `/help /tools /memory /memory search /forget /clear /status /vision /exit`.
  - `Start-NEXUS.bat` → `app/launcher/menu.py` — cloud-only menu: `[1] Start NEXUS (NVIDIA cloud)`, `[2] Health Check`, `[3] Exit`. Starts no servers, kills no processes.
  - `app/main.py:build_assistant()` — provider + registry + executor + memory factory. `get_nexus_info()` resolves NVIDIA hosted values for display.
- Core: `app/brain/assistant.py` (`NexusAssistant.achat/chat`, `achat_with_image/chat_with_image`, `_achat_with_tools` agent loop), `app/brain/provider.py` (`EchoProvider`, `OpenAICompatibleProvider`, `NvidiaNemotronProvider`, `resolve_nvidia_settings`, `resolve_nvidia_vision_settings`, `nvidia_vision_stale_override`), `app/brain/prompts.py`, `app/brain/tasks.py` (new, untracked — `TaskTracker`/`StepRecord`), `app/brain/planner.py` + `router.py` (diagnostic Phase 1 stubs, still wired: `planner.plan()` is logged per turn; `router.route()` covers the tools-disabled legacy path — **not dead**).
- Tools: `app/tools/` — 20 tools when `WEB_ENABLED=true` (12 always: 4 system + 5 filesystem + applications.launch + terminal.execute + brave.open; +8 web/browser when enabled). Single enforcement point `ToolExecutor` (validate → assess → policy → confirm → timeout → audit). Schemas/wire mapping in `app/tools/schemas.py` (incl. Gemini thought-signature preservation).
- Memory: `app/memory/` — local SQLite (`data/nexus.db` by default), `MemoryManager`, redaction (refuse/redact secrets), per-conversation scoping.
- Voice: `app/voice/` — `WhisperSTT` (default per `config.py`) + `VoskSTT` fallback, `SapiTTS`/`NullTTS`, `SoundDeviceRecorder`, `VoiceSession`. Local-only audio; confirmations auto-deny in voice.
- Vision: `app/vision/loader.py` (Pillow confine/verify/resize/EXIF-strip) + `describer.py` (`OpenRouterVisionDescriber` endpoint-agnostic + `MetadataDescriber` offline) + `factories.py` (`is_cloud_vision`, NVIDIA/local/hosted branches). Explicit `/vision <path> [question]` upload only; history/memory store basename+text, never bytes.
- Web/browser: `app/web/` (`client.py` search+fetch, `ssrf.py` fail-closed per-hop checks, `extract.py` stdlib parsing, `search_session.py` latest-results session) + `app/browser/session.py` (reader sessions) + `app/tools/web.py|browser.py|brave.py`.
- Security: `app/security/policy.py` (SAFE/LOW/MEDIUM/HIGH/BLOCKED), `confirmation.py` (console/auto-deny), `audit.py` (`sanitize` + `AuditLogger` → `logs/tools.jsonl`).
- Config: `app/core/config.py` (`Settings`, `get_settings` cached, `model_dump_safe` redacts 3 keys). Text model default `nvidia/nemotron-3.5-lightning-30b-a3b`, vision default `meta/llama-3.2-11b-vision-instruct`, base `https://integrate.api.nvidia.com/v1`. Tool loop: `MAX_TOOL_ITERATIONS=3`, `TOOL_LOOP_TIMEOUT_SECONDS=300`, `TOOL_RESULT_MAX_CHARS=20000`.
- Uncommitted working-tree changes (16 modified + 4 untracked): NVIDIA provider/vision, Phase 7 loop (dedup, truncation, timeout, tracker, prompts, synthesis turn), cloud launcher, health/vision display, `nvapi` audit pattern, executor unexpected-exception guard, `docs/TOOLS.md` Phase 7 section. Untracked: `app/brain/tasks.py`, `tests/test_nvidia_provider.py`, `tests/test_nvidia_vision.py`, `tests/test_phase7_orchestration.py`.

## 2. Active capabilities and implementations (verified by code + passing tests)

- NVIDIA Nemotron text chat (`NvidiaNemotronProvider`, flattened `temperature/top_p/max_tokens/chat_template_kwargs/reasoning_budget`, non-streaming; `health_detail()` categories; resolvers ignore loopback base URLs and GGUF-path model leftovers).
- NVIDIA cloud vision (`resolve_nvidia_vision_settings`, retired-ID ignore for `nvidia/nemotron-nano-12b-v2-vl`, loopback refusal, explicit missing-key errors; `/vision` disclosure line shows basename + resolved model).
- Phase 7 orchestration (`MAX_TOOL_ITERATIONS` turns, wall-clock budget with first turn always running, per-request dedup via name+args JSON, `TOOL_RESULT_MAX_CHARS` truncation marker, progress notes, tool-free synthesis turn, honest budget-exhausted fallback with `tracker.summary()`, empty-turn guard).
- System/filesystem/terminal/applications tools with confinement, allowlists, timeouts, caps, confirmations, audit.
- Memory (SQLite, redaction, deterministic recall, CLI `/memory /forget /status`).
- Web search (keyless DDG HTML default + legacy Instant Answer JSON branch), page fetch, reader browser sessions, visible `brave.open` (LOW, no approval, argv `shell=False`).
- Voice (Whisper default + Vosk fallback, SAPI/Null TTS, scripted headless checks, auto-deny approvals).
- Safety confirmations (`Allow? [y/N]`, auto-deny non-interactive/broken callback), auditing with `sanitize` (incl. `nvapi` tokens), event-bus payloads sanitized, provider/vision/tool/memory failures honest without credential leaks.
- Launcher (cloud-only menu + read-only health rows: venv, NVIDIA text/vision config presence, local-servers NOT USED, DB, voice/vision/web flags; `run.py --health` live check + vision key-presence block that uploads nothing).

## 3. Confirmed findings (evidence-backed)

1. **Docs describe superseded configurations.** `README.md` (Groq `openai/gpt-oss-120b`, `openrouter/free` vision, `echo/openai` provider table, Phase 6 "awaiting release", Phase 5.0 structure), `docs/VISION.md` (`openrouter/free`, shared `AI_API_KEY`, no NVIDIA), `docs/VOICE.md` ("memory + tools + Groq"), `docs/WEB.md` (`WEB_SEARCH_URL=https://api.duckduckgo.com/` vs code default `https://html.duckduckgo.com/html/`; web example mentions Groq lineup), `docs/LOCAL.md` (full local Qwen/llama.cpp manual as if normal operation). Code reality is NVIDIA cloud normal + local/legacy branches retained.
2. **Version/phase markers stale.** `app/__init__.py`: docstring Phase 5.0, `__phase__ = 5`. `app/main.py:get_nexus_info()` returns `"phase": 6`. Working tree implements Phase 7 orchestration (uncommitted). `app/launcher/__init__.py` docstring says "local AI launcher ... ensures a model server is up" — false for cloud `menu.py`.
3. **Config/doc default mismatches.** `config.py` defaults vs `.env.example`: `vision_provider` default `openai-compatible` vs example `nvidia`; `voice_stt_backend` default `whisper` vs example `vosk`; `voice_language` default `en` vs example `en-US`; `voice_stt_model_path` default `small.en` vs example `data/vosk-model-small-en-us-0.15`; `vision_model` default `openrouter/free` (placeholder the NVIDIA resolver ignores) vs example empty. `docs/WEB.md` search-URL default contradicts `config.py`. `client.py` module docstring says Instant Answer API only, but code supports HTML default + legacy JSON.
4. **Unreachable duplicate handler in `run.py`.** `cmd_health()` has two consecutive identical `except Exception as exc` blocks after one `try` (lines ~228–235); the second can never run.
5. **`requirements.txt` lists `fastapi`/`uvicorn` with zero repo-wide usage.** `grep` for `fastapi|uvicorn|FastAPI` hits only `requirements.txt` (2 lines). No imports in `app/`, `tests/`, `scripts/`. Confirmed unused today; `README` reserves API/dashboard for "later phases".
6. **Generated/ignored artifacts present but correctly ignored.** `__pycache__/`, `.pytest_cache/`, `.venv/`, `data/nexus.db`, `data/vosk-model-small-en-us-0.15/`, `logs/tools.jsonl`, `logs/servers/text.log` (0 bytes) exist on disk and are git-ignored (`.env`, `data/*.db`, `logs/*.log|jsonl`, `data/vosk-model-*`, `.venv/`). `git ls-files` confirms `.env` untracked. Nothing secret is tracked.
7. **`.env` (local, untracked) has only 15 keys vs ~50 in `.env.example`.** Missing keys fall back to code defaults (memory/web/tool-loop/voice-tuning). Not an error, but health/config docs should state defaults apply. Values never printed by this audit.
8. **Retired-model handling is correct in code, stale in user files.** Code ignores `nvidia/nemotron-nano-12b-v2-vl` (retired 2026-08-26) and surfaces a "remove it" note in health. `.env.example` documents this; any user `.env` still pinning it gets the safe default (verified by `test_launcher.py` + resolver tests).

## 4. Suspected items — NEED manual confirmation (NOT confirmed, NOT deleted)

- `switch-model.ps1` (tracked, local llama.cpp text/vision switcher). No code imports it; launcher no longer uses it. **Hypothesis:** legacy local helper, superseded for cloud users. **Retain:** local `servers.py`/`vision_provider=local` paths and their tests still exist, so legacy local users may still use it. Needs owner decision.
- `app/launcher/servers.py` + `handoff.py` (llama-server lifecycle + vision→text switch). `menu.py` no longer imports them; cloud `run_cloud()` attaches no handoff. **Hypothesis:** dead in cloud path. **Retain:** `vision_provider=local`, `.env.example` local sections, `docs/LOCAL.md`, and tests (`test_launcher.py` server unit tests, `test_vision_handoff.py`, `test_vision_local.py`) still exercise them; deleting breaks backward compat and passing tests.
- `docs/LOCAL.md` as a whole. **Hypothesis:** contradicts cloud-only normal operation. **Retain:** it documents the legacy local branches that still exist in code. Fix is a scope header, not deletion.
- `vision-test.jpg` (tracked, ~612 KB). Only referenced by `docs/LOCAL.md` manual steps; no test imports it. **Hypothesis:** manual `/vision` fixture. **Retain:** deleting user-facing test assets on a guess risks breaking manual verification; needs owner decision.
- `frontend/dashboard/.gitkeep`, `scripts/.gitkeep` placeholders. **Retain:** intentional phase placeholders, negligible cost.
- `planner.py`/`router.py` look like stubs but are wired (see §1). **Retain — not dead.**
- `test_local_provider.py`, `test_vision_local.py`, `test_vision_openrouter.py`, `test_gemini_compat.py` cover legacy/compat paths. **Retain:** they pass and guard backward compat.

## 5. Dependency and configuration issues

- `fastapi>=0.115`, `uvicorn>=0.30`: confirmed unused (see §3.5). Proposed: remove from `requirements.txt` (no code change needed; future API phase can re-add with actual imports). Low risk; keeps install surface minimal.
- Everything else in `requirements.txt` is used (lazy or direct): `httpx` (provider/vision/web/servers), `pydantic`/`pydantic-settings`/`python-dotenv` (config), `sqlalchemy` (memory), `pytest`/`pytest-asyncio` (suite), `pyttsx3`/`vosk`/`sounddevice`/`numpy` (voice), `faster-whisper` (Whisper STT), `av<19` (faster-whisper dep), `Pillow` (vision loader). No blind upgrades proposed.
- `.gitignore` covers `__pycache__/`, `*.py[cod]`, `.venv/`, `venv/`, `.env`, `data/*.db|sqlite*|vosk-model-*|*.zip`, `logs/*.log|jsonl`, caches, `.env.backup-local-model`, `logs/servers/*.log`. Gaps: `.pytest_cache/` covered; `data/*.db-journal|wal|shm` (SQLite sidecars) and `logs/servers/` dir itself are not explicitly covered — minor hardening worth adding. Must not ignore source/docs/tests.
- `.env.example` voice block disagrees with `config.py` voice defaults (see §3.3); `VISION_PROVIDER=nvidia` in example disagrees with code default `openai-compatible` (cloud vision still works because `is_cloud_vision()` keys off `AI_PROVIDER=nvidia`, but the two sources of truth confuse readers). Needs one documented story.
- `run.py` duplicate `except` (see §3.4) — remove the dead block, no behavior change.

## 6. Documentation inaccuracies (to fix in Stage 4)

- `README.md`: Groq/OpenRouter/echo-openai framing, Phase 6 "awaiting release", Phase 5.0 structure listing (omits `launcher/servers|handoff`, `brain/tasks.py`, NVIDIA provider, Phase 7 limits, `NVIDIA_*` vars), stale config tables.
- `docs/VISION.md`: OpenRouter-only; missing NVIDIA cloud path, retired-ID handling, `VISION_PROVIDER=nvidia`, `NVIDIA_VISION_MODEL`.
- `docs/VOICE.md`: Groq reference; voice defaults table should match `config.py` + `.env.example` after reconciliation.
- `docs/WEB.md`: `WEB_SEARCH_URL` default wrong; provider-setup section should lead with HTML default, legacy JSON second.
- `docs/LOCAL.md`: needs a "legacy local operation, not normal NVIDIA cloud operation" scope header; launcher menu excerpt inside it (options 1–4 local) contradicts current 3-option cloud menu.
- `docs/TOOLS.md`: already updated for Phase 7 in working tree — keep, verify limits match `config.py`.
- `docs/MEMORY.md`: largely accurate; keep, minor touch-ups only.
- `.env.example`: Groq/local examples read as current options; NVIDIA block says "vision-capable Nemotron Nano VL model" while the pinned default is `meta/llama-3.2-11b-vision-instruct` — wording fix needed. Also reconcile voice/vision defaults.
- `app/__init__.py`, `app/launcher/__init__.py`, `app/web/client.py` docstrings (see §3.2–3.3).

## 7. Security and reliability findings

- Good: `sanitize()` covers `nvapi` tokens (new), Bearer patterns, sensitive keys; `model_dump_safe()` redacts all three keys; audit/event-bus payloads sanitized; `.env` untracked; health/vision checks upload nothing and print no secrets/images; vision logs strip base64; memory refuses/redacts secrets; SSRF fail-closed per hop; terminal `shell=False` + confinement + confirmations; executor converts unexpected tool exceptions to audited failures (new) so one bad tool cannot kill the loop.
- Watch items (no change without owner need): regex command classification is best-effort (documented honestly in `docs/TOOLS.md`); `voice_tts_volume=1.0` + `rate=175` are permissive but local-only; `TOOL_LOOP_TIMEOUT_SECONDS=300` is generous — intentional for multi-step research, documented; no new secret patterns observed beyond `nvapi` (covered).
- Secret scan: tracked diff (`git diff --name-only`) contains no `.env`, no DB, no logs, no keys — only source/docs/tests/launcher. No credentials printed in this report.

## 8. Proposed cleanup actions and reasons (safe subset only)

1. Remove duplicate `except` in `run.py:cmd_health` (dead code, no behavior change).
2. Refresh stale docstrings: `app/__init__.py` (phase/version), `app/launcher/__init__.py` (cloud + legacy servers note), `app/web/client.py` (HTML default + legacy JSON).
3. Remove `fastapi`/`uvicorn` from `requirements.txt` (confirmed zero usage; re-add when API phase lands).
4. Harden `.gitignore` with SQLite sidecars (`data/*.db-journal`, `data/*.wal`, `data/*.shm`); keep everything else.
5. Reconcile `.env.example` ↔ `config.py` defaults (voice backend/language/model-path, `VISION_PROVIDER`, search URL) with NVIDIA-as-normal + legacy-labeled Groq/local examples; fix Nano-VL wording to match pinned `meta/llama-3.2-11b-vision-instruct`.
6. Rewrite `README.md` + `docs/VISION.md`/`VOICE.md`/`WEB.md` scope/headers + `docs/LOCAL.md` legacy header + config tables to match code (model IDs, base URL, env names, commands, limits, approval/timeout/iteration behavior, quotas/errors, persistence, OS constraints, phase status from code+tests).
7. No source/test deletions. No dependency upgrades. No local-inference reintroduction. No commits/pushes.

AI-wellness note: this audit deliberately keeps legacy local paths and their tests intact; cloud-only means "normal operation", not "deleted fallback code".

## 9. Actions completed (2026-10-13, uncommitted — awaiting review)

- `run.py`: removed unreachable duplicate `except` in `cmd_health` (dead code, no behavior change); header/flag text Phase 6 → 7.
- `app/__init__.py`: docstring + `__phase__` 5 → 7 to match working-tree orchestration. `app/main.py`: `get_nexus_info()["phase"]` 6 → 7. No test asserts the value.
- `app/launcher/__init__.py`: docstring corrected (cloud menu + legacy servers/handoff note). `app/web/client.py`: module docstring corrected (HTML default + legacy JSON).
- `requirements.txt`: removed confirmed-unused `fastapi`/`uvicorn` (zero repo-wide imports; re-add when the API phase lands). No upgrades.
- `.gitignore`: added SQLite sidecars (`data/*.db-journal`, `data/*.wal`, `data/*.shm`). No source/docs/tests hidden.
- `.env.example`: NVIDIA vision wording fixed (Nano VL → pinned `meta/llama-3.2-11b-vision-instruct`); Groq + local Qwen + local vision examples labeled legacy; voice block notes code defaults (`whisper`/`small.en`/`en`) vs pinned Vosk fallback; vision block notes `VISION_PROVIDER` default vs explicit `nvidia` pin. No real credentials.
- `README.md`: rewritten to NVIDIA cloud reality (model IDs, base URL, env names, key setup, start/health/test/voice/vision/web/memory/tools, permissions/approvals/timeouts/limits, quotas/errors with no free-use promise, persistence/deps/Windows constraints, Phase 7 status, legacy-local pointer).
- `docs/VISION.md`: rewritten to NVIDIA cloud (resolution precedence, retired-ID handling, synthesis modes, config table). `docs/VOICE.md`: Groq reference removed; whisper-default vs Vosk-example clarified. `docs/WEB.md`: search-URL default corrected to `https://html.duckduckgo.com/html/`, Groq example → NVIDIA. `docs/LOCAL.md`: legacy-scope header + superseded-menu note.
- Intentionally retained (see §4): `servers.py`, `handoff.py`, `switch-model.ps1`, `docs/LOCAL.md` body, `vision-test.jpg`, `frontend/dashboard` + `scripts` placeholders, `planner.py`/`router.py`, all legacy/compat tests. No source/test files deleted.
- Verification: full suite rerun after changes (see Stage 6); `git diff --check` clean; secret scan of tracked files clean; `.env` untracked; launcher verified cloud-only (no `servers`/process imports in `menu.py`).
- Baseline → final: 621 passed / 1 skipped → **621 passed / 1 skipped** (rerun 2026-10-13, ~27s; no regressions). Targeted subset (config/launcher/NVIDIA/vision/phase7/tool-loop/audit/web/voice): 154 passed. `compileall` clean. `git diff --check` clean. Live `run.py --health` (single cheap call, no quota chat, no image upload): NVIDIA text resolved correctly, vision `configured` (key set), provider `timeout` on this run — reported honestly, no secrets in output, no retries.

## 10. NVIDIA-only correction (2026-10-13, uncommitted — awaiting review)

Supersedes the retain-legacy stance in §4/§8-§9: local model operation was
removed outright per explicit direction. NVIDIA (`https://integrate.api.nvidia.com/v1`,
`nvidia/nemotron-3.5-lightning-30b-a3b`, `meta/llama-3.2-11b-vision-instruct`)
is the single AI path; failures raise, never fall back.

- `.env`: removed the duplicate `AI_TIMEOUT_SECONDS=180` (lines 13+23 held
  the identical value; 16→15 active settings, everything else byte-identical).
  Within-file duplicates resolve last-wins via python-dotenv; OS environment
  overrides `.env` (both covered by new tests). Request timeout demonstrably
  flows `Settings.ai_timeout_seconds` → provider/vision HTTP clients; the
  health probe uses its own short 5s budget, now named in timeout details so
  probe timeouts are never confused with request timeouts.
- Deleted: `app/launcher/servers.py`, `app/launcher/handoff.py`
  (llama-server lifecycle + vision→text local switch; no production imports),
  `switch-model.ps1` (local model switcher; referenced nowhere),
  `tests/test_vision_handoff.py`, `tests/test_vision_local.py` (subjects
  removed). Renamed `tests/test_local_provider.py` →
  `tests/test_provider_transport.py` (mocked transport tests, loopback URLs
  replaced with `https://mock.test`).
- `NexusAssistant` + `build_assistant`: `synthesis_handoff` hook removed;
  vision synthesis goes to the NVIDIA text model or falls back to the direct
  report honestly.
- Voice: `WhisperSTT`/`VoskSTT` removed from `app/voice/stt.py` (protocol +
  `ScriptedSTT` remain); `create_stt` raises `VoiceUnavailableError` for any
  live backend with text-mode/`--script` guidance. SAPI TTS retained and
  labeled as a local Windows OS utility (not NVIDIA, not AI inference); mic
  capture retained as device I/O; nothing uploaded. `faster-whisper`, `av`,
  `vosk` removed from `requirements.txt` (kept: `pyttsx3`, `sounddevice`,
  `numpy`, `Pillow`, all still used).
- Vision: `vision_provider=local` now raises an explicit use-NVIDIA error;
  `_create_local_describer` removed (`is_local_url` kept as the NVIDIA
  loopback guard). Generic hosted `openai-compatible` vision retained as an
  explicit cloud option; `GroqVisionDescriber` alias retained for compat.
- `EchoProvider` retained as the offline non-AI test/dev stub (documented as
  such); `OpenAICompatibleProvider` retained as the NVIDIA transport base.
  No silent fallback exists on any NVIDIA path (verified by tests).
- Voice capability audit: no NVIDIA STT/TTS today. Verification of a cloud
  speech endpoint was attempted (two live web searches) but the search
  transport failed, and no endpoint/model/format is referenced anywhere in
  the implementation — so none was invented. Live mic STT is an honest,
  explicit unavailable state; this is reported as a limitation, not worked
  around. If NVIDIA publishes a verified speech API later, it slots into the
  `SpeechToText`/`TextToSpeech` protocols (see `docs/VOICE.md`).
- Docs: `docs/LOCAL.md` is now a removal/migration notice; `docs/VOICE.md`
  rewritten (capability table, unavailable STT, local-OS TTS labeling);
  `README.md`/`.env.example`/`docs/VISION.md` updated (Groq + local examples
  deleted, not merely labeled); `docs/WEB.md` states retrieval≠AI and NVIDIA
  does all reasoning/synthesis.
- Tests: new `tests/test_nvidia_only.py` (18 tests: single-provider
  selection, no-fallback failures, no-handoff plumbing, leftover-ignoring
  resolvers, local-vision refusal, settings-driven timeout, probe-budget
  wording, key-presence≠healthy, dotenv last-wins + OS precedence, voice
  STT refusal + script/TTS working + text-mode isolation, web-retrieval →
  NVIDIA-synthesis citation, key redaction incl. `nvapi`); `test_launcher.py`
  rewritten (menu behavior + absence of server modules/switcher script);
  `test_voice_backends.py` rewritten to unavailable-state assertions.
- Full suite after correction: **602 passed, 1 skipped** (was 621; delta =
  removed local-server/handoff/local-vision tests, net of +18 new). No
  regressions; `compileall` and `git diff --check` clean. User data
  (`data/nexus.db`, `data/vosk-model-*`, `logs/`, `.venv/`, `.env`, ignored
  backups) untouched. Nothing committed or pushed.
