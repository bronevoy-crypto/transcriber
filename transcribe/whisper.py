"""Faster-Whisper транскрибер."""
import time

import numpy as np
import structlog

from transcribe.base import BaseTranscriber, TranscriptionResult

logger = structlog.get_logger(__name__)

_SAMPLE_RATE = 16000


class WhisperTranscriber(BaseTranscriber):
    def __init__(self, config: dict):
        self._model_name = config.get("model", "large-v3")
        self._language = config.get("language", "ru")
        self._device = config.get("device", "cuda")
        self._compute_type = config.get("compute_type", "float16")
        self._model = None

    def load(self) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError("Установите faster-whisper: pip install faster-whisper")

        logger.info("WhisperTranscriber: загрузка модели...", model=self._model_name, device=self._device)
        self._model = WhisperModel(
            self._model_name,
            device=self._device,
            compute_type=self._compute_type,
        )
        logger.info("WhisperTranscriber: модель загружена")

    def is_loaded(self) -> bool:
        return self._model is not None

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        if not self.is_loaded():
            raise RuntimeError("WhisperTranscriber: модель не загружена")

        audio_float = audio.astype(np.float32) / 32768.0

        t0 = time.monotonic()
        segments, _ = self._model.transcribe(
            audio_float,
            language=self._language,
            beam_size=5,
            vad_filter=False,  # используем собственный VAD
        )
        text = " ".join(seg.text.strip() for seg in segments)
        elapsed_ms = (time.monotonic() - t0) * 1000

        logger.debug("WhisperTranscriber: decoded", text=text[:60], latency_ms=round(elapsed_ms))
        return TranscriptionResult(text=text, confidence=0.9 if text else 0.0)
