# NEXUS Voice (Phase 8: NVIDIA cloud speech)

Speech input and output around the existing `NexusAssistant`. Live voice
mode is fully functional: microphone audio → NVIDIA cloud transcription →
NVIDIA Nemotron reply (tools + memory) → spoken response.

| Capability | Implementation | AI inference? | State |
|---|---|---|---|
| Speech-to-text (live mic) | NVIDIA cloud Parakeet CTC 1.1B English (`nvidia/parakeet-ctc-1_1b-asr`) via NVCF HTTPS | Yes (NVIDIA cloud) | **Works** (`--voice`) |
| Speech-to-text (headless) | `ScriptedSTT` canned transcripts | No (replays text) | Works via `--script` |
| Text-to-speech (default) | Windows SAPI via pyttsx3 | No (local OS synthesizer) | Works on Windows |
| Text-to-speech (cloud) | NVIDIA Magpie TTS (`nvidia/magpie-tts-multilingual`) via NVCF HTTPS | Yes (NVIDIA cloud) | Works with `VOICE_TTS_BACKEND=nvidia` |
| Silent output | `NullTTS` | No | Works (tests, quiet runs) |
| Microphone capture | `sounddevice` PortAudio | No (device I/O) | Works (only in live voice mode) |
| Reasoning/memory/tools | NVIDIA cloud text model | Yes (NVIDIA cloud) | Works |

**Verified NVIDIA speech services (live HTTP 200, Oct 2026, same
`NVIDIA_API_KEY` as chat):**

- STT model `nvidia/parakeet-ctc-1_1b-asr` (English):
  `POST https://1598d209-5e27-4d3c-8079-4751568b1081.invocation.api.nvcf.nvidia.com/v1/audio/transcriptions`
  with `Authorization: Bearer <nvapi-...>` and multipart
  `file=<WAV 16-bit mono>` + `language=en-US` → `{"text": "..."}`.
  Silence returns `{"text": ""}` (skipped, session keeps listening).
  Round-trip verified: Magpie-generated "Hello from NEXUS voice test."
  transcribes back verbatim. Docs: `https://build.nvidia.com/nvidia/parakeet-ctc-1_1b-asr/api`
  and `https://docs.nvidia.com/nim/speech/latest/reference/api-references/asr/http-asr.html`.
- TTS model `nvidia/magpie-tts-multilingual`:
  `POST https://877104f7-e885-42b9-8de8-f6e4c6303969.invocation.api.nvcf.nvidia.com/v1/audio/synthesize`
  with multipart `text + language=en-US + voice=Magpie-Multilingual.EN-US.Aria`
  → `audio/wav`. Voices listed via `GET .../v1/audio/list_voices`.
  Docs: `https://build.nvidia.com/nvidia/magpie-tts-multilingual/api`.

No local AI speech models are used. Requesting `whisper`/`vosk` raises an
explicit error pointing to `VOICE_STT_BACKEND=nvidia`. There is deliberately
no silent fallback: STT failures raise per-utterance (session continues),
missing credentials refuse startup with guidance (exit 2).

Wispr Flow / WhisperFlow investigation: the installed dictation app's vendor
(Wispr Flow) does publish a Voice Interface API (WebSocket + REST at
`platform-api.wisprflow.ai`, base64 16 kHz WAV, Bearer key, exclusive
partnership approval required per `https://api-docs.wisprflow.ai`). It was
not used: a verified NVIDIA cloud STT already satisfies the NVIDIA-only
requirement, and adding a non-NVIDIA transcription vendor would violate
"NVIDIA remains the provider". No Wispr integration was added.

The Windows SAPI voice is an ordinary operating-system utility, like
Notepad or the terminal. It is **not** NVIDIA speech generation and is
never presented as such. `VOICE_TTS_BACKEND=nvidia` is the only path that
uses NVIDIA speech generation.

## Architecture

```
microphone -> AudioRecorder -> capture_utterance (energy VAD)
  -> pcm_to_wav_bytes -> NvidiaSTT.transcribe (NVCF HTTPS)
  -> text -> NexusAssistant.chat() (Nemotron + tools + memory)
  -> reply -> TTS.speak (+ console print)
```

- `app/voice/audio.py` — `AudioRecorder` protocol, `SoundDeviceRecorder`
  (PortAudio), `capture_utterance()` silence-bounded recording, `chunk_rms()`,
  `pcm_to_wav_bytes()` (PCM → WAV in memory for upload).
- `app/voice/stt.py` — `SpeechToText` protocol, `NvidiaSTT` (verified cloud
  Parakeet; httpx multipart, timeouts/auth/rate-limit mapped to `STTError`),
  `ScriptedSTT` (headless/scripted). Local-AI recognizers do not exist.
- `app/voice/tts.py` — `TextToSpeech` protocol, `SapiTTS` (local Windows
  OS synthesizer), `NvidiaTTS` (verified cloud Magpie; WAV played locally
  via sounddevice), `NullTTS` (silent sink).
- `app/voice/session.py` — `VoiceSession`: one turn = listen, transcribe,
  skip empties, stop-phrase check, `assistant.chat()`, speak. No assistant
  logic duplicated; confirmation-required tools stay auto-denied in voice
  mode (approve them in text mode).
- `app/voice/factories.py` — backend selection from settings
  (`resolve_nvidia_stt_settings` / `resolve_nvidia_tts_settings`).
- `app/voice/errors.py` — `VoiceUnavailableError`,
  `MicrophoneUnavailableError`, `STTUnavailableError`, `STTError`,
  `TTSUnavailableError`, `TTSError`, `VoiceCancelledError`.

## Running voice mode

```powershell
.venv\Scripts\python.exe run.py --voice
.venv\Scripts\python.exe run.py --voice --script "hello | what time is it"  # no mic needed
VOICE_TTS_BACKEND=null .venv\Scripts\python.exe run.py --voice --script "status"  # silent
VOICE_TTS_BACKEND=nvidia .venv\Scripts\python.exe run.py --voice  # cloud voice replies
```

Live mode captures the microphone only after explicit `--voice` startup.
Say a stop phrase (`stop listening`, `goodbye nexus`, `exit voice`) or press
Ctrl+C to exit. Each utterance is one bounded recording (leading silence
skipped, trailing-silence stop, max-utterance cap); empty/silent captures
are skipped without calling the assistant, so there are no duplicate
actions and no repeated uploads of the same audio.

## Interruption ("wait" / "continue")

Live microphone mode (English only) listens while a reply is playing, using
the same already-open microphone stream — playback uses the speaker device,
so there are no competing microphone consumers:

- Say **“wait”** (also “hold on”, “pause”, “stop”, “quiet”) to stop playback
  promptly and pause. Say **“continue”** (also “resume”, “go on”) to hear
  the remaining sentences. Only short utterances (at most 6 words) count as
  controls; a longer sentence containing “wait” is answered as a new
  question, so corrections are never swallowed.
- Asking a new question mid-answer stops the old reply and answers the new
  turn immediately. An undecipherable interruption (silence, noise, STT
  error) resumes the remaining reply instead of punishing the user.
- Resume granularity is per sentence: fully played sentences are never
  repeated, but the sentence cut off mid-play replays from its start.
- Replies are split into sentences for playback (over-long sentences split
  on word boundaries to respect the 2000-character Magpie request limit).
  `--script` headless mode stays strictly sequential.
- Echo handling is energy-based only (no neural model, no echo canceller):
  silence never triggers anything, and only loud sustained speech is even
  transcribed. Headphones or lower speaker volume reduce self-triggering.

## Configuration (.env)

`VOICE_ENABLED`, `VOICE_STT_BACKEND` (`nvidia` + parakeet aliases; legacy
`whisper`/`vosk` refused with migration guidance), `VOICE_TTS_BACKEND`
(`sapi` = local Windows OS synthesizer | `nvidia` = NVIDIA cloud Magpie |
`null` = silent), `VOICE_SAMPLE_RATE` (16000), `VOICE_CHANNELS` (1),
`VOICE_SILENCE_TIMEOUT_SECONDS` (1.2), `VOICE_MAX_UTTERANCE_SECONDS` (15),
`VOICE_TTS_RATE` (175), `VOICE_TTS_VOLUME` (1.0), `VOICE_STOP_PHRASES`,
`VOICE_STT_LANGUAGE` (en-US), `VOICE_STT_TIMEOUT_SECONDS` (30),
`NVIDIA_STT_FUNCTION_ID`, `NVIDIA_STT_BASE_URL` (empty = derived),
`VOICE_TTS_LANGUAGE` (en-US), `VOICE_TTS_VOICE`
(`Magpie-Multilingual.EN-US.Aria`), `VOICE_TTS_SAMPLE_RATE_HZ` (44100),
`VOICE_TTS_TIMEOUT_SECONDS` (30), `NVIDIA_TTS_FUNCTION_ID`,
`NVIDIA_TTS_BASE_URL` (empty = derived).
Legacy `VOICE_STT_MODEL_PATH` / `VOICE_LANGUAGE` / device keys are inert.
Speech uses `NVIDIA_API_KEY` (`AI_API_KEY` fallback), never logged.
See `.env.example`.

## Troubleshooting

- `Speech recognition backend 'whisper' was removed` → set
  `VOICE_STT_BACKEND=nvidia` (verified cloud STT).
- `NVIDIA_API_KEY is not set` → set `NVIDIA_API_KEY=nvapi-...` from
  `https://build.nvidia.com` in `.env`; or use text mode / `--script`.
- `Speech service authentication failed (HTTP 401/403)` → key invalid or
  revoked; check `build.nvidia.com` key, never paste it into logs.
- `Speech service rate-limited (HTTP 429)` / `quota/payment (402)` → wait
  or check NVIDIA account; the session keeps listening.
- `Speech service timed out` → network slow; raise
  `VOICE_STT_TIMEOUT_SECONDS` or retry.
- `Speech service rejected the audio (HTTP 400)` → capture produced
  invalid audio; retry speaking clearly.
- `sounddevice is not installed` → `pip install sounddevice numpy`
  (microphone device access + cloud TTS playback).
- `Microphone: UNAVAILABLE` → enable an input device in Windows Sound
  settings; close apps exclusively holding the mic.
- `Speech output unavailable` (SAPI) → Windows Settings → Time & language
  → Speech; or use `nvidia` / `null` TTS.
- `Voice mode is disabled` → `VOICE_ENABLED=false` blocks even `--script`.

## Privacy behavior

- Live voice mode sends microphone audio to NVIDIA cloud STT
  (`<function-id>.invocation.api.nvcf.nvidia.com` over HTTPS) for
  transcription, and (with `VOICE_TTS_BACKEND=nvidia`) sends reply text to
  NVIDIA cloud TTS. Transcripts go to the NVIDIA chat model as the user
  message, like typed text.
- Audio is captured to RAM only while a live session runs, encoded to WAV
  in memory, sent once per utterance, then dropped. Nothing is recorded to
  disk; recordings are not retained.
- Raw audio and API keys are never logged. Transcriptions are not written
  to audit logs (only tool executions are audited, as in text mode).
- Scripted checks (`--script`) use canned text, not audio; nothing leaves
  the machine.

## Limitations

- STT is English (`en-US`) via the verified Parakeet model; other
  languages need a different verified function ID (not configured).
- TTS cloud text limit is 2000 chars per utterance (excess truncated).
- Spoken output by default is OS-quality SAPI synthesis, not a neural
  cloud voice (use `VOICE_TTS_BACKEND=nvidia` for Magpie).
- Energy-based utterance detection (no neural VAD) governs recording
  bounds; interruption controls are English word-boundary matches on short
  utterances only (see above).
- Resume replays from the interrupted sentence (sentence granularity).
- No barge-in in `--script` headless mode (strictly sequential).
- Confirmation-required (MEDIUM+) tools are denied in voice mode — use
  text mode to approve them.
- Single-user, single-session: one microphone, one conversation.

## Replacing backends

`NvidiaSTT` / `NvidiaTTS` implement the `SpeechToText` / `TextToSpeech`
protocol (`is_available`, `transcribe` / `speak` + `synthesize`, `stop`).
New verified NVIDIA speech IDs register in `factories` + config with mocked
tests — the session and assistant need no changes. Do not reintroduce
local AI models as a stand-in.
