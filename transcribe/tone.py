"""T-one (T-Tech / voicekit-team) — русский CTC для телефонии, 8 kHz CPU."""
import threading
import time

import numpy as np
import structlog

from transcribe.base import BaseTranscriber, TranscriptionResult
from transcribe.factory import register

logger = structlog.get_logger(__name__)


def _resample_16k_to_8k(audio_int16: np.ndarray) -> np.ndarray:
    # T-one ждёт int32 @ 8 kHz, наш пайплайн отдаёт int16 @ 16 kHz.
    # Ресэмпл делаем через scipy.signal.resample_poly (FIR-фильтр, без артефактов).
    from scipy.signal import resample_poly
    audio_f = audio_int16.astype(np.float32)
    audio_8k = resample_poly(audio_f, up=1, down=2)
    return audio_8k.astype(np.int32)


@register("tone")
class ToneTranscriber(BaseTranscriber):
    # T-one отдаёт только phrase-level разбиение (по тишине), пословных
    # таймингов нет — диаризация падает в fallback speaker_at по сегменту.
    supports_word_timestamps = False

    def __init__(self, config: dict):
        self._config = config
        self._pipeline = None
        self._lock = threading.Lock()

    def load(self) -> None:
        try:
            from tone import StreamingCTCPipeline, DecoderType
        except ImportError:
            raise ImportError("Установите T-one: pip install tone")

        logger.info("ToneTranscriber: загрузка модели с HuggingFace...")
        # DecoderType.GREEDY не требует KenLM (бинарные биндинги не собираются
        # на Windows без C++ toolchain). BEAM_SEARCH даёт WER чуть ниже, но
        # зависит от kenlm — если он установлен, можно переключить в конфиге.
        decoder_name = (self._config.get("decoder") or "greedy").lower()
        decoder = DecoderType.BEAM_SEARCH if decoder_name == "beam_search" else DecoderType.GREEDY
        self._pipeline = StreamingCTCPipeline.from_hugging_face(decoder_type=decoder)
        logger.info("ToneTranscriber: модель загружена", decoder=decoder.value)

    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        if not self.is_loaded():
            raise RuntimeError("ToneTranscriber: модель не загружена")

        audio_int32_8k = _resample_16k_to_8k(audio)

        t0 = time.monotonic()
        with self._lock:
            phrases = self._pipeline.forward_offline(audio_int32_8k)

        text = " ".join(p.text for p in phrases if p.text).strip()
        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.debug("ToneTranscriber: decoded", text=text[:60], latency_ms=round(elapsed_ms))

        return TranscriptionResult(text=text)
