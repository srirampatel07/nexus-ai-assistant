# NEXUS Jarvis Parity (Phase 8 — complete)

All three Jarvis tracks, mapped to NEXUS tools. Single provider rule holds:
chat/vision/voice inference stays NVIDIA cloud; connectors use keyless or
env PATs, never stored in memory DB or logs. 46 tools total.

## Track 1 — getjarvis.eu (screen-aware)
- `screen.capture` (`app/screen/capture.py` + `app/tools/screen.py`): explicit
  desktop JPEG to `data/screenshots/` for `/vision`. STRICT privacy disables it.
- `screen.point`: Tkinter Show-Me circle at (x,y) + label, auto-close.
- CLI: `/screen` captures; then `/vision "<path>" <question>`.
- `connectors.status`: 15 specs (github/gmail/calendar/slack/notion/linear/
  jira/trello/todoist/asana/discord/obsidian/drive/spotify/weather) — presence
  only, values never shown. Credentials via env (see `.env.example`).
- `github.repo` / `github.issues`: first live connectors (keyless public,
  `GITHUB_TOKEN` for private).
- `slack.search` / `notion.search` / `jira.search` / `todoist.list`: read-only,
  env-gated (honest missing-var errors, no values logged).
- `research.report`: 2-search evidence pack for deep-research reports
  (model writes the cited report; combine with `web.fetch`).
- Personas/tones (`app/personas/catalog.py`): 6 personas
  (assistant/tutor/coach/interviewer/writer/debugger) + 5 tones.
  CLI `/persona`, `/tone`; `NEXUS_PERSONA`/`NEXUS_TONE` shape the system prompt.
- Drag-drop = copy file into allowed roots, then `/vision <path>`.

## Track 2 — jarvis.institute (dev / agentic)
- Think-act-observe loop already Phase 7; Phase 8 adds smart pre-filter
  (`app/brain/budget.py:select_relevant_tools`, `SMART_TOOLS_*`) + token
  guards (`BudgetGuard`, `NEXUS_SESSION_TOKEN_CAP`/`NEXUS_REPLY_TOKEN_CAP`).
- `git.status/log/diff/commit` (confined, commit needs approval).
- `docker.ps/images/logs/start/stop` (mutating needs approval).
- `routines.schedule/list/due/delete` (SQLite `data/routines.db`, min 60s,
  catch-up via `routines.due`, CLI `run.py --routines-due`).
- `reminders.create/list` (SQLite `data/reminders.db`).
- Command Center: `run.py --briefing` + `/briefing` (`app/server/briefing.py`)
  + `frontend/dashboard/index.html` skeleton (memory counts, routines due,
  reminders open, connectors configured).
- `run.py --serve` hosts the skeleton + `/api/briefing` on 127.0.0.1:8765
  (`app/server/api.py`, stdlib only).
- `run.py --routines-run` executes due routines via the assistant with
  catch-up (`app/routines/runner.py`).
- Privacy modes `NEXUS_PRIVACY_MODE=strict/guarded/relaxed`: strict disables
  `screen.capture` today; web/image callouts documented for next pass.
- Proactive nudge = poll `routines.due` + briefing (full background daemon roadmap).

## Track 3 — jarvis.ceo (voice-first)
- Voice stays NVIDIA Parakeet STT + SAPI/Magpie TTS with RMS interrupt
  (`VOICE_INTERRUPT_RMS_THRESHOLD`); personas/tones apply to replies.
- `VOICE_STT_LANGUAGE`/`VOICE_TTS_LANGUAGE` expose multi-lang config
  (en-US verified; other langs best-effort via Magpie).
- `image.generate` (keyless Pollinations → `data/images/`).
- Reminders/calendar, deep-research (= Phase 7 multi-search loop + cite URLs),
  open-apps (`applications.launch` allowlist) cover the rest; call/SMS,
  Siri, mobile client remain out of scope (no phone radio on desktop).

## Privacy / honesty deltas vs Jarvis claims
- NEXUS: local SQLite, `logs/tools.jsonl` redacted, allowed-roots confinement,
  MEDIUM+ approval. No training. But NVIDIA cloud = text/audio/images leave
  the box; no EU-residency promise. STRICT mode is local-gating, not E2E crypto.
- getjarvis.eu AES-256-GCM/GDPR/EU-residency and ceo no-telemetry are hosted
  claims NEXUS does not make.

## Try it
```
python run.py --briefing
python run.py --routines-due
python run.py --routines-run
python run.py --serve   # agent console: http://127.0.0.1:8765/
/tools → screen.capture, git.*, docker.*, routines.*, reminders.*, connectors.*, image.generate
/screen → /vision "<screenshot>" what do you see?
/persona tutor → /tone direct → /personas
research with cites: "write a deep report on X" (research.report + web.fetch)
/personas, /briefing, /routines-due in text mode
```

## Agent console (dashboard controls everything)
- `run.py --serve` hosts the agent console (localhost only, stdlib
  `ThreadingHTTPServer`, `app/server/api.py`): chat, tool runner, screen
  capture, vision, persona/tone, routines-run, live stats — no CLI needed.
- `GET /api/briefing`, `GET /api/tools`; `POST /api/chat {text, approve?}`,
  `/api/tool/run {name, arguments?, approve?}`, `/api/screen/capture`,
  `/api/vision {path, question?}`, `/api/persona {persona?, tone?}`,
  `/api/routines/run`. MEDIUM+ tools auto-deny unless `approve:true`
  (dashboard auto-approve toggle, default off). Bodies capped 32KB.
- `frontend/dashboard/index.html` is the agent-style UI: center chat,
  quick actions (git/connectors/screen/due/reminders/deep report), tool
  runner with risk labels, vision box, persona pickers, live due/reminder/
  connector/system panels, 15s auto-stats.
- Mic 🎤: browser SpeechRecognition (Chrome/Edge) dictates into chat;
  speaker toggle reads replies via speechSynthesis. No server change.
- Paste/drag image: Ctrl+V or 📎 in chat uploads to `data/screenshots/`
  (`POST /api/upload {filename, data_b64}`, 10MB cap, PNG/JPG/WEBP) then
  auto-runs vision. Chat history in localStorage + Export/Clear, Esc stops.
- Gmail, direct from dashboard: Connect panel sets `GMAIL_ADDRESS` +
  `GMAIL_APP_PASSWORD` and persists to `.env` (survives restarts; every
  server loads them at startup).
  `gmail.inbox` (IMAP unread + snippets + uids), `gmail.read` (full body by
  uid), `gmail.send` (SMTP, HIGH + approval), `gmail.reply` (uid + quoted
  reply, HIGH + approval). Needs Google 2-Step app password.
- WhatsApp: Connect panel sets `WHATSAPP_TOKEN` + `WHATSAPP_PHONE_NUMBER_ID`
  (Meta Cloud API); `whatsapp.send` (HIGH + approval) or `whatsapp.draft`
  (SAFE, copy-paste, no creds needed).
