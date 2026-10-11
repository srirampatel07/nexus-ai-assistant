# NEXUS fully local: Qwen3-8B text + Qwen2.5-VL-3B vision (llama.cpp)

## One-click launcher (Windows)

Double-click **`Start-NEXUS.bat`** in the project root — no manual venv
activation, no manual servers:

```
NEXUS — LOCAL AI ASSISTANT

[1] Text Mode — Qwen3-8B
[2] Vision Mode — Qwen2.5-VL-3B
[3] Health Check
[4] Exit

Select an option:
```

- **[1]** stops the launcher-owned vision server if running, starts text
  on :8080, waits for API readiness, then opens the normal text UI.
- **[2]** stops the launcher-owned text server if running, starts vision
  (+ mmproj) on :8081, waits for readiness, then opens the normal text
  UI where `/vision <image-path> [question]` works. (There is no
  `--vision` flag by design — vision lives inside the text interface.)
- **[3]** prints a read-only health report (Python/venv, llama-server,
  model + projector files, endpoint identity vs `.env`, memory DB,
  voice/vision/web flags, RAM, port occupancy). Configured is never
  reported as operational: endpoints show DOWN/WARN unless the API
  answers with the expected model.
- **[4]** / Ctrl+C stops launcher-owned servers only, then exits. The
  menu loops until you exit; leaving a mode returns to it.

**Safety rules the launcher enforces:** loopback bind only; one large
model at a time (16 GB); a port occupant is stopped only with pidfile +
command-line proof (anything else is reported as a conflict and left
alone); a running server is reused only after `/v1/models` identity
proof; readiness is polled with bounded timeouts and failures clean up
spawned processes; logs at `logs/servers/text.log` and `vision.log`;
3.5 GB free-RAM gate before starting; no downloads, no installs, no
cloud fallback.

NEXUS talks to local models through its existing `openai-compatible`
stack — no new dependency, no hosted key, no Ollama. Text runs on
`Qwen3-8B-Q4_K_M.gguf` (:8080); vision runs on `Qwen2.5-VL-3B-Instruct`
(`ggml-org` GGUF + matching mmproj) on **:8081**, configured independently
via `VISION_*` settings.

## 1. Start the model server

```powershell
$model = "$env:USERPROFILE\.cache\huggingface\hub\models--Qwen--Qwen3-8B-GGUF\snapshots\7c41481f57cb95916b40956ab2f0b139b296d974\Qwen3-8B-Q4_K_M.gguf"
$server = (Get-Command llama-server.exe).Source
& $server -m $model --host 127.0.0.1 --port 8080 -c 4096 -t 6 -b 256
```

Leave that window running. Do not change `-c`/`-t` unless you know why.

## 2. Verify the model id

```powershell
Invoke-RestMethod http://127.0.0.1:8080/v1/models | ConvertTo-Json -Depth 6
```

Note the exact `id` (currently the full GGUF path). `AI_MODEL` must match
it exactly.

## 3. Configure NEXUS (`.env` — your file, edit these four lines)

```dotenv
AI_PROVIDER=openai-compatible
AI_BASE_URL=http://127.0.0.1:8080/v1
AI_MODEL=C:\Users\shash\.cache\huggingface\hub\models--Qwen--Qwen3-8B-GGUF\snapshots\7c41481f57cb95916b40956ab2f0b139b296d974\Qwen3-8B-Q4_K_M.gguf
AI_API_KEY=
```

`AI_API_KEY` stays empty: local mode sends no `Authorization` header.
Everything else in `.env` stays as-is. Hosted configs still work — just
swap these four lines back.

## 4. Health check and chat test

```powershell
.\.venv\Scripts\python.exe run.py --health
.\.venv\Scripts\python.exe run.py --text
> hello — reply with exactly: local-ok
```

`--health` should report `"healthy": true` with the local base URL and
model. Tool calling works against this server (verified: the model
returns real `tool_calls`, executed through the normal permission +
audit pipeline).

## 5. Troubleshooting (text server)

| Symptom | Cause / fix |
|---|---|
| `AI provider unreachable … ConnectError` | Server not running — start it (§1). |
| Health `false` but server runs | Wrong port — `AI_BASE_URL` port must match `--port`. |
| `Address already in use` from llama-server | Port 8080 taken — stop the other process or use another `--port` and match `AI_BASE_URL` to it. |
| `HTTP 404 … model` | `AI_MODEL` mismatch — re-check §2 for the exact id. |
| Slow replies | Expected on CPU; keep prompts short. Context is 4096 tokens. |
| `tools not supported` fallback answers | Server build without tool support — NEXUS answers in plain text honestly. |

## 6. Local vision server (Qwen2.5-VL-3B, port 8081)

Files (one-time download, ~2.6 GB total, already in your HF cache):
`Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf` (1.8 GB) +
`mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf` (0.8 GB), repo
`ggml-org/Qwen2.5-VL-3B-Instruct-GGUF`.

```powershell
$vl = "$env:USERPROFILE\.cache\huggingface\hub\models--ggml-org--Qwen2.5-VL-3B-Instruct-GGUF\snapshots\5037fcf163dd95d1e41d1974465f0898ed108ca2\Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf"
$mm = "$env:USERPROFILE\.cache\huggingface\hub\models--ggml-org--Qwen2.5-VL-3B-Instruct-GGUF\snapshots\5037fcf163dd95d1e41d1974465f0898ed108ca2\mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf"
$server = (Get-Command llama-server.exe).Source
& $server -m $vl --mmproj $mm --alias qwen2.5-vl-3b-local --host 127.0.0.1 --port 8081 -c 4096 -t 6 -b 256
```

Verify: `Invoke-RestMethod http://127.0.0.1:8081/v1/models` should list
id `qwen2.5-vl-3b-local` (the `--alias` keeps config stable across
snapshot updates).

## 7. Configure NEXUS for local vision (`.env` — your file)

```dotenv
VISION_ENABLED=true
VISION_PROVIDER=local
VISION_BASE_URL=http://127.0.0.1:8081/v1
VISION_MODEL=qwen2.5-vl-3b-local
VISION_API_KEY=
```

`VISION_PROVIDER=local` refuses any non-loopback endpoint and sends no
credential — images can never reach a hosted service in this mode, and a
missing base/model raises an explicit error instead of silently
downgrading. Text `AI_*` settings stay exactly as they are.

## 8. Test image analysis

```powershell
.\.venv\Scripts\python.exe run.py --text
> /vision vision-test.jpg What do you see?
```

Expect a real description grounded in the image (the CLI prints which
model analyzed it). Failures are explicit: server down → "unreachable";
wrong id → HTTP 404; misconfigured local mode → exact missing setting.

## 10. Vision + text handoff (one model at a time)

`/vision vision-test.jpg What do you see in this image?` runs this
lifecycle (16 GB RAM: the two large models are never resident together):

1. Validate the image (allowed roots, type/size caps, EXIF strip).
2. Analyze on :8081 (Qwen2.5-VL-3B) and validate the response.
3. If :8080 (Qwen3-8B) answers, it synthesizes the final answer from the
   vision report — the reply is authored by the text model, not pasted.
4. If :8080 is down and a launcher handoff is attached (menu Vision
   Mode), NEXUS **once** stops the owned :8081 server, starts :8080,
   waits for readiness, then synthesizes. End state is predictable:
   **TEXT server running** — pick Vision Mode again for another image.
5. If the switch is unsafe or fails (unowned occupant, low RAM, startup
   timeout, text inference error), the vision report is shown directly
   with a fallback notice naming the failure. Synthesis is never claimed
   unless Qwen3 actually generated the answer.

`VISION_TEXT_SYNTHESIS=false` skips synthesis entirely (raw report, zero
text-model contact). Without a handoff attached (plain `run.py --text`),
the previous direct-report fallback applies unchanged.

### Manual integration procedure (real models, ~10 min, mocked tests excluded)

```powershell
# 0. Text server down, vision server up (or use menu option [2]).
# 1. From the Vision Mode text UI:
> /vision vision-test.jpg What do you see in this image?
# 2. Expect (after a model switch + synthesis pause): a natural answer
#    authored from the vision findings — NOT the raw-report prefix
#    ("My text brain is offline…").
# 3. Confirm end state: :8080 serving Qwen3-8B, :8081 free:
> /status
netstat -ano -p TCP | Select-String ':8080|:8081'
# 4. Failure drill (optional): stop :8080 mid-way or set
#    VISION_TEXT_SYNTHESIS=false — expect the direct report, honestly labeled.
```

## 9. Troubleshooting (vision server)

| Symptom | Cause / fix |
|---|---|
| `VISION_BASE_URL is not set` | `VISION_PROVIDER=local` needs an explicit loopback URL (§7). |
| `refuses non-loopback endpoint` | Local mode only allows 127.0.0.1/localhost/::1 — by design. |
| `HTTP 404 … model` on :8081 | `VISION_MODEL` must match the server `--alias` (or file id). |
| `Missing projector / mtmd` at server start | `--mmproj` path wrong — re-check §6 paths. |
| `Vision model timed out` after ~30 s | CPU vision is slow: first image analysis can take minutes. Raise `AI_TIMEOUT_SECONDS` (e.g. 180) — it caps text and vision requests alike. |
| Sluggish vision on 16 GB RAM | Normal on iGPU/CPU; run the :8081 server only when analyzing images, and close it afterwards. Measured with both servers resident: ~5 GB free of 15.7 GB — workable, not roomy. |

No Ollama, no hosted provider involved.
