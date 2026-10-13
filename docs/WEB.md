# NEXUS Web + Browser (Phase 6)

Web search, page retrieval, and document-navigation browser sessions —
all through the standard Phase 2 tool pipeline (registry → executor →
permissions → confirmation → audit). Retrieval is plain HTTP + parsing
(DuckDuckGo results pages, static page text) — it is not AI. All
reasoning over retrieved content (comparison, summarization, answer
generation) is done by the NVIDIA cloud text model through the normal
assistant tool loop, with source URLs cited. No new network dependencies
(`httpx` + stdlib only). No paid APIs, no additional external services.

## Architecture

```
User -> NexusAssistant -> ToolExecutor -> web.*/browser.* tools
                                                    |
                         app/web/  (shared HTTP layer)
                           ssrf.py    per-hop SSRF validation
                           extract.py stdlib readable-text extraction
                           client.py  search + fetch (injectable transport/DNS)
                         app/browser/ (stateful reader sessions)
                           session.py ReaderSession + BrowserManager
Tools: app/tools/web.py (web.search, web.fetch)
       app/tools/browser.py (browser.navigate/read/links/click/back/close)
       app/tools/brave.py (brave.open — visible desktop Brave)
Sessions: app/web/search_session.py (latest web.search results for "result N")
```

## Capabilities

| Tool | Risk | Approval | What it does |
|---|---|---|---|
| `web.search` | LOW | none | Keyless web search → titles, URLs, snippets |
| `web.fetch` | LOW | none | One page → title, readable text, links, final URL |
| `browser.navigate` | LOW | none | Open a URL in a named session |
| `browser.read` | SAFE | none | Current page title + text (cached, no refetch) |
| `browser.links` | SAFE | none | Current page links with stable ids |
| `browser.click` | MEDIUM | **yes** | Follow a link id, or open a URL |
| `browser.back` | SAFE | none | Previous page (cached) |
| `browser.close` | SAFE | none | Drop a session and its history |
| `brave.open` | LOW | none | Any public URL / `result N` → visible Brave window |

The browser is a **document reader**: links (anchors) are the only
supported interactive elements. Forms, script-backed buttons, logins,
downloads, payments, and JavaScript are **not** supported — requests for
them are reported as unsupported, never faked.

## Dependencies and configuration

No new packages. Network uses the existing `httpx`; DNS checks use
stdlib `socket`; HTML parsing uses stdlib `html.parser`.

`.env` (all optional, offline-safe defaults; no secrets — the default
search endpoint is keyless):

```
WEB_ENABLED=true
WEB_SEARCH_PROVIDER=duckduckgo
WEB_SEARCH_URL=https://html.duckduckgo.com/html/
WEB_SEARCH_TIMEOUT=15
WEB_SEARCH_MAX_RESULTS=8
WEB_FETCH_TIMEOUT=20
WEB_FETCH_MAX_BYTES=1048576
WEB_FETCH_MAX_REDIRECTS=5
WEB_TEXT_MAX_CHARS=20000
```

`WEB_ENABLED=false` removes the eight `web.*`/`browser.*` tools; `brave.open`
is independent of this flag (desktop capability, like `applications.launch`).
A custom `WEB_SEARCH_URL` pointing at `https://api.duckduckgo.com/` uses the
legacy Instant Answer JSON shape (`AbstractText`/`RelatedTopics`).

## Provider setup

Default: nothing to configure — queries go to the keyless DuckDuckGo
HTML results page (`https://html.duckduckgo.com/html/?q=…`, verified
live during development), parsed for real result titles, destination
URLs (unwrapped from redirect links), and snippets. A legacy
DuckDuckGo Instant Answer JSON endpoint still parses when configured
via `WEB_SEARCH_URL`. If the provider blocks us, rate-limits, or
changes its markup, search fails honestly with a diagnostic (or an
empty result set) instead of fabricated results.

## Running with web capabilities

```powershell
python run.py --text
> search the web for the current NVIDIA Nemotron model lineup
> fetch https://example.com and summarize it
> open https://example.com in the browser, list its links, follow link 1
> search for the latest ESP32 documentation
> open result 2 in Brave
> open https://docs.espressif.com/projects/esp32/en/latest/ in Brave
```

The brain selects `web.*`/`browser.*`/`brave.*` tools itself;
`browser.click` pauses for `Allow? [y/N]` (`brave.open` launches validated
URLs automatically). Voice mode works unchanged (speech routes through
normal chat; approvals stay auto-deny there).

## Visible Brave browser (`brave.open`)

Opens any valid public HTTP/HTTPS destination in your real, visible
Brave desktop window — a separate operation from the internal reader
sessions above (those never open desktop windows).

* **Any domain, no allowlist.** Pass a full URL (`https://…`, path/query/
  fragment preserved exactly), a bare hostname (`python.org/doc` → opened
  as `https://`), or a current-search reference: `result 2`, `2`, `#3`,
  `second`, `the second result`, `first`, `last`. References resolve
  against the latest successful `web.search` output only — real result
  URLs, never reconstructed or guessed. A failed or empty search clears
  the session, so stale results can never be opened by accident; vague
  references (`that website`) resolve only when exactly one result exists,
  otherwise NEXUS asks which result you mean.
* **Discovery.** Brave is located via the Windows App Paths registry,
  standard install directories (`Program Files`, `Program Files (x86)`,
  `LocalAppData`), then `PATH`. A missing Brave (or a launch failure)
  is reported honestly — NEXUS never claims a page loaded merely because
  a process started (the result says *started*, with pid).
* **Validation, not SSRF-gating.** `brave.open` checks URL syntax only:
  http/https scheme (`file:`/`javascript:`/`data:`/`ftp:` refused),
  hostname present, no embedded `user:pass@` credentials, sane length.
  Opening a URL in *your* browser is client-side navigation, so the
  server-side SSRF fetch protections (private-IP/DNS rules) do not apply
  here — they stay fully enforced on `web.fetch`/`browser.*`.
* **Approvals + audit.** LOW risk with no confirmation: validated website
  launches run automatically (like launching an allowlisted app); invalid
  targets are still rejected by validation before anything starts. Audit
  logs the canonical `brave.open` name with sanitized arguments. Launched
  via argv (`shell=False`) — no shell ever sees the URL.
* **No redundant re-fetch.** The success result tells the model the page is
  visible in Brave and not to re-read it with `web.fetch`/`browser.*`
  unless content was explicitly requested — so open-only requests finish
  inside the normal tool budget instead of burning turns on duplicate
  retrievals.

## Permissions and approvals

* Read-only operations (`search`, `fetch`, `navigate`, `read`, `links`,
  `back`, `close`) run immediately — same as file reads.
* `browser.click` is MEDIUM + `requires_confirmation`: interactive CLI
  asks, headless/voice contexts auto-deny. (`brave.open` is LOW with no
  confirmation — plain page opening only. Anything that submits forms,
  sends messages, purchases, or modifies accounts keeps its existing
  confirmation gates.)
* Restricted destinations are refused at `assess()` time, audited as
  `denied:policy` — before any connection.

## Security limitations

* **All web content is untrusted.** The system prompt requires citing
  source URLs, never following page instructions, and never claiming
  unverified actions. Tool results are evidence, not assistant speech.
* **SSRF guard** (`app/web/ssrf.py`, fail-closed): http/https only, no
  credentials in URLs, no single-label/local/special-use names, no
  non-global IPs (incl. multicast), DNS must resolve with **every**
  returned address routable. Applied to the initial URL **and each
  redirect hop** (max 5), before connecting.
* **Bounds**: 1 MB default response cap, 20000-char text cap, per-tool
  executor timeouts. Non-readable types (images, binaries) are refused
  without retention. Query strings are preserved to the origin server
  (standard HTTP) but never logged with credentials — audit sanitizing
  applies to all tool arguments.
* **Known unsupported**: JavaScript rendering, authentication, form
  submission, file downloads, POSTing data, arbitrary element clicking.
  Link clicks are plain GET navigations; approve them only when the
  destination is expected.

## Tests

```powershell
python -m pytest tests/test_web_search.py tests/test_web_fetch.py tests/test_web_ssrf.py tests/test_browser.py tests/test_web_integration.py tests/test_brave_open.py -q
python -m pytest -q   # full suite, still fully mocked/offline
```

Coverage: valid/invalid queries, structured results, 429/500/timeout/
malformed/empty responses, extraction (markup/comments/scripts dropped,
entities, relative links), invalid URLs, redirects (chain, off-site,
restricted-target, loops), oversized bodies, unsupported types, SSRF
matrix (literals, DNS-to-private, multi-record, resolution failure,
userinfo, schemes), browser lifecycle + approval + cleanup, brain
tool-selection + citations + failure honesty, prompt-injection
non-escalation, audit redaction, disabled-mode registry, and
chat/memory/voice/vision regression smoke tests.

## Manual verification

1. `python run.py --text`, ask for recent/step-known info requiring the
   web; confirm the assistant calls `web.search`/`web.fetch` and cites
   URLs (visible with `--debug` history + `logs/tools.jsonl`).
2. `browser.navigate` a docs page, `browser.links`, `browser.click`
   with `y` then `N` — confirm navigation vs. denial behavior.
3. `web.fetch http://169.254.169.254/` — confirm refusal, no connection.
4. `WEB_ENABLED=false python run.py --text` — confirm `/tools` omits
   `web.*`/`browser.*` (but still lists `brave.open`) and chat is unaffected.
5. Search for any topic, then `open result N in Brave` — Brave opens the
   exact result URL with no approval prompt; try an out-of-range number
   and confirm an honest "could not resolve" message with nothing opened.
