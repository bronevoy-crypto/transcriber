"""T-one (T-Tech / voicekit-team) — русский CTC для телефонии, 8 kHz CPU."""
import threading
import time
from pathlib import Path

import numpy as np
import structlog

from transcribe.base import BaseTranscriber, TranscriptionResult
from transcribe.factory import register

_MODELS_DIR = Path(__file__).parent.parent / "models" / "tone"

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
            print("T-one: пакет не установлен, устанавливаю...")
            import subprocess, sys
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "--no-deps",
                    "git+https://github.com/voicekit-team/T-one.git",
                ])
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", "pyctcdecode"])
                subprocess.check_call([sys.executable, "-m", "pip", "install", "pygtrie"])
                from tone import StreamingCTCPipeline, DecoderType
            except Exception:
                raise ImportError(
                    "T-one не удалось установить автоматически.\n"
                    "Установите вручную:\n"
                    "  pip install --no-deps git+https://github.com/voicekit-team/T-one.git\n"
                    "  pip install pyctcdecode\n"
                    "Требуется git: https://git-scm.com/download/win"
                )

        decoder_name = (self._config.get("decoder") or "greedy").lower()
        decoder = DecoderType.BEAM_SEARCH if decoder_name == "beam_search" else DecoderType.GREEDY

        # Скачиваем модель в models/tone/ если ещё нет
        model_file = _MODELS_DIR / "model.onnx"
        if not model_file.exists():
            logger.info("ToneTranscriber: скачивание модели в models/tone/...")
            from huggingface_hub import hf_hub_download
            _MODELS_DIR.mkdir(parents=True, exist_ok=True)
            hf_hub_download(
                repo_id="t-tech/T-one",
                filename="model.onnx",
                local_dir=str(_MODELS_DIR),
            )

        logger.info("ToneTranscriber: загрузка модели...", decoder=decoder.value)
        self._pipeline = StreamingCTCPipeline.from_local(_MODELS_DIR, decoder_type=decoder)
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
