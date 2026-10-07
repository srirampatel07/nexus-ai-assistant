"""Speech-to-text abstraction (Phase 4).

Production STT can use local Faster-Whisper or offline Vosk.
ScriptedSTT is used by tests/headless runs.

Audio is processed in memory and is never stored by this module.
"""

from __future__ import annotations

import io
import json
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from app.voice.errors import STTError, STTUnavailableError, VoiceCancelledError

VOSK_MODEL_DOWNLOAD = (
    "https://alphacephei.com/vosk/models/"
    "vosk-model-small-en-us-0.15.zip"
)


@dataclass
class TranscriptionResult:
    text: str = ""
    confidence: float | None = None
    duration_seconds: float | None = None


@runtime_checkable
class SpeechToText(Protocol):
    """PCM (16-bit mono) in, transcription out."""

    def is_available(self) -> tuple[bool, str]:
        """(usable, detail) probe without transcribing."""
        ...

    def transcribe(self, pcm: bytes, sample_rate: int) -> TranscriptionResult:
        """Transcribe one utterance."""
        ...


class WhisperSTT:
    """Local Faster-Whisper speech recognizer.

    No audio is uploaded to a server.
    The model is downloaded once and then runs locally.
    """

    def __init__(
        self,
        model_path: str = "small.en",
        language: str = "en",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        self.model_path = model_path
        self.language = language
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def _load(self):  # type: ignore[no-untyped-def]
        if self._model is not None:
            return self._model

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise STTUnavailableError(
                "faster-whisper is not installed. Run: "
                "uv pip install faster-whisper"
            ) from exc

        try:
            self._model = WhisperModel(
                self.model_path,
                device=self.device,
                compute_type=self.compute_type,
            )
        except Exception as exc:
            raise STTUnavailableError(
                f"Cannot load Whisper model '{self.model_path}': {exc}"
            ) from exc

        return self._model

    def is_available(self) -> tuple[bool, str]:
        try:
            self._load()
        except STTUnavailableError as exc:
            return False, str(exc)
        except Exception as exc:
            return False, f"Whisper unavailable: {exc}"

        return True, (
            f"faster-whisper model: {self.model_path}, "
            f"device={self.device}, compute_type={self.compute_type}"
        )

    @staticmethod
    def _pcm_to_wav(pcm: bytes, sample_rate: int) -> io.BytesIO:
        wav_buffer = io.BytesIO()

        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm)

        wav_buffer.seek(0)
        return wav_buffer

    def transcribe(
        self,
        pcm: bytes,
        sample_rate: int,
    ) -> TranscriptionResult:
        if not pcm:
            return TranscriptionResult()

        if sample_rate <= 0:
            raise STTError("Invalid sample rate.")

        try:
            model = self._load()

            wav_buffer = self._pcm_to_wav(pcm, sample_rate)

            segments, info = model.transcribe(
                wav_buffer,
                language=self.language,
                beam_size=5,
                best_of=5,
                temperature=0.0,
                vad_filter=True,
                condition_on_previous_text=False,
            )

            collected = []
            probabilities = []
            duration = 0.0

            for segment in segments:
                text = (segment.text or "").strip()

                if text:
                    collected.append(text)

                probability = getattr(
                    segment,
                    "avg_logprob",
                    None,
                )

                if isinstance(probability, (int, float)):
                    # Convert Whisper's log probability into a rough
                    # human-readable confidence value.
                    import math

                    confidence = math.exp(float(probability))
                    confidence = max(0.0, min(1.0, confidence))
                    probabilities.append(confidence)

                end = getattr(segment, "end", None)
                if isinstance(end, (int, float)):
                    duration = max(duration, float(end))

            text = " ".join(collected).strip()

            confidence = (
                sum(probabilities) / len(probabilities)
                if probabilities
                else None
            )

            if duration <= 0:
                duration = len(pcm) / 2 / sample_rate

            return TranscriptionResult(
                text=text,
                confidence=confidence,
                duration_seconds=duration,
            )

        except STTUnavailableError:
            raise
        except Exception as exc:
            raise STTError(f"Whisper transcription failed: {exc}") from exc


class VoskSTT:
    """Offline Vosk recognizer.

    Kept as a fallback backend for compatibility with the original
    Phase 4 implementation.
    """

    def __init__(self, model_path: str, language: str = "en-US") -> None:
        self.model_path = Path(model_path).expanduser()
        self.language = language
        self._recognizer = None

    def _load(self):  # type: ignore[no-untyped-def]
        try:
            from vosk import KaldiRecognizer, Model
        except ImportError as exc:
            raise STTUnavailableError(
                "vosk is not installed. Install it with: "
                "uv pip install vosk and download a model."
            ) from exc

        if not self.model_path.is_dir() or not any(
            self.model_path.iterdir()
        ):
            raise STTUnavailableError(
                f"Vosk model not found at '{self.model_path}'. "
                f"Download {VOSK_MODEL_DOWNLOAD}, unzip it there, and retry."
            )

        try:
            return KaldiRecognizer(
                Model(str(self.model_path)),
                16000,
            )
        except Exception as exc:
            raise STTUnavailableError(
                f"Cannot load Vosk model: {exc}"
            ) from exc

    def is_available(self) -> tuple[bool, str]:
        try:
            self._load()
        except STTUnavailableError as exc:
            return False, str(exc)

        return True, f"vosk model: {self.model_path}"

    def transcribe(
        self,
        pcm: bytes,
        sample_rate: int,
    ) -> TranscriptionResult:
        if not pcm:
            return TranscriptionResult()

        try:
            recognizer = self._load()

            try:
                recognizer.SetWords(True)
            except Exception:
                pass

            block = max(1, sample_rate // 4)

            for offset in range(0, len(pcm), block):
                recognizer.AcceptWaveform(
                    pcm[offset : offset + block]
                )

            final = json.loads(
                recognizer.FinalResult() or "{}"
            )

        except STTUnavailableError:
            raise
        except Exception as exc:
            raise STTError(
                f"Vosk transcription failed: {exc}"
            ) from exc

        text = str(final.get("text") or "").strip()

        words = final.get("result") or []

        confs = [
            word.get("conf")
            for word in words
            if isinstance(word.get("conf"), (int, float))
        ]

        confidence = (
            sum(confs) / len(confs)
            if confs
            else None
        )

        return TranscriptionResult(
            text=text,
            confidence=confidence,
            duration_seconds=(
                len(pcm) / 2 / sample_rate
                if sample_rate
                else None
            ),
        )


class ScriptedSTT:
    """Dev/headless backend: replays canned transcripts in order."""

    def __init__(self, transcripts: list[str]) -> None:
        self._transcripts = list(transcripts)

    def is_available(self) -> tuple[bool, str]:
        return True, (
            f"scripted ({len(self._transcripts)} utterance(s) queued)"
        )

    def transcribe(
        self,
        pcm: bytes,
        sample_rate: int,
    ) -> TranscriptionResult:
        if not self._transcripts:
            raise VoiceCancelledError(
                "Script exhausted; ending voice session."
            )

        return TranscriptionResult(
            text=self._transcripts.pop(0)
        )