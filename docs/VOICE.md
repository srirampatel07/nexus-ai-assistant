# NEXUS Voice Interface (Phase 4)

Hands-free interface around the existing `NexusAssistant`: microphone
captures speech, offline STT transcribes it, the normal assistant pipeline
(memory + tools + Groq) answers, and local TTS speaks the reply.

## Architecture

```
microphone -> AudioRecorder -> capture_utterance (energy VAD)
  -> STT.transcribe -> text -> NexusAssistant.chat()
  -> reply -> TTS.speak (+ console print)
```

- `app/voice/audio.py` — `AudioRecorder` protocol, `SoundDeviceRecorder`
  (PortAudio), `capture_utterance()` silence-bounded recording, `chunk_rms()`.
- `app/voice/stt.py` — `SpeechToText` protocol, `VoskSTT` (offline),
  `ScriptedSTT` (headless/scripted).
- `app/voice/tts.py` — `TextToSpeech` protocol, `SapiTTS` (Windows SAPI),
  `NullTTS` (silent sink).
- `app/voice/session.py` — `VoiceSession`: one turn = listen, transcribe,
  skip empties, stop-phrase check, `assistant.chat()`, speak. No assistant
  logic duplicated; confirmation-required tools stay auto-denied in voice
  mode (approve them in text mode).
- `app/voice/factories.py` — backend selection from settings.
- `app/voice/errors.py` — `VoiceUnavailableError`,
  `MicrophoneUnavailableError`, `STTUnavailableError`, `STTError`,
  `TTSUnavailableError`, `TTSError`, `VoiceCancelledError`.

## Supported backends

| Direction | Backend | Engine | Network | Cost |
|---|---|---|---|---|
| STT | `vosk` (default) | Vosk small English model, local | none | free |
| STT | `script` (via `--script`) | canned text, no mic | none | free |
| TTS | `sapi` (default) | Windows SAPI via pyttsx3 | none | free |
| TTS | `null` | silent sink (logs text) | none | free |

No cloud STT, no extra LLM, no paid API.

## Windows setup

```powershell
.venv\Scripts\activate
pip install -r requirements.txt   # pyttsx3, vosk, sounddevice, numpy
```

Download the STT model (~40 MB, one time):

1. Get `vosk-model-small-en-us-0.15.zip` from
   https://alphacephei.com/vosk/models/
2. Unzip into `data/vosk-model-small-en-us-0.15/` (so `am/`, `graph/` …
   sit directly inside). `data/vosk-model-*` is git-ignored.

Requires: a working microphone (input device) and speakers. The
PortAudio library ships inside the `sounddevice` wheel — no separate
driver install on standard Windows.

## Microphone / speaker requirements

- An enabled recording device (check Windows Sound settings → Input).
- The app requests no special permissions itself; if the OS denies
  microphone access, startup reports `Microphone: UNAVAILABLE` and exits.
- Speakers/headphones for `sapi`; use `VOICE_TTS_BACKEND=null` for silent.

## Configuration (.env)

`VOICE_ENABLED, VOICE_STT_BACKEND (vosk), VOICE_TTS_BACKEND (sapi|null),
VOICE_STT_MODEL_PATH, VOICE_SAMPLE_RATE (16000), VOICE_CHANNELS (1),
VOICE_LANGUAGE, VOICE_SILENCE_TIMEOUT_SECONDS (1.2),
VOICE_MAX_UTTERANCE_SECONDS (15), VOICE_TTS_RATE (175),
VOICE_TTS_VOLUME (1.0), VOICE_STOP_PHRASES
(stop listening,goodbye nexus,exit voice)`. See `.env.example`.

## Running voice mode

```powershell
python run.py --voice
python run.py --voice --script "hello | what time is it"  # no mic needed
VOICE_TTS_BACKEND=null python run.py --voice --script "status"  # silent
```

Startup prints STT/TTS/microphone/speaker status (never secrets), then
`Listening...`. Exit with a stop phrase or Ctrl+C — the microphone stream
is always released.

## Troubleshooting

- `sounddevice is not installed` → `pip install sounddevice numpy`.
- `Vosk model not found` → download + unzip as above; check
  `VOICE_STT_MODEL_PATH`.
- `Microphone: UNAVAILABLE` → enable an input device in Windows Sound
  settings; close apps exclusively holding the mic.
- `Speech output unavailable` → SAPI voices missing; Windows Settings →
  Time & language → Speech; or use `null` TTS.
- Transcription is poor → speak clearly, reduce background noise, keep
  utterances under ~15 s.

## Privacy behavior

- Audio is captured to RAM only and handed to the local STT engine.
- Nothing is recorded to disk; recordings are not retained.
- Raw audio is never logged. Transcriptions are not written to audit logs
  (only tool executions are audited, as in text mode).
- No microphone audio leaves the machine (both default backends are local).

## Limitations

- Energy-based utterance detection (no neural VAD): very quiet speech or
  loud rooms may cut early/late; tune `VOICE_SILENCE_TIMEOUT_SECONDS`.
- Small Vosk model: good for commands, weaker on rare words.
- No barge-in: NEXUS finishes speaking before listening again.
- Confirmation-required (MEDIUM+) tools are denied in voice mode — use
  text mode to approve them.
- Single-user, single-session: one microphone, one conversation.

## Replacing backends

Implement the `SpeechToText` / `TextToSpeech` protocol
(`is_available`, `transcribe` / `speak`, `stop`), register the name in
`factories.create_stt/create_tts`, add offline tests with fakes — the
session and assistant need no changes.
