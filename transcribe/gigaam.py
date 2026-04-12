"""GigaAM v3 транскрибер через sherpa-onnx (CTC или RNNT)."""
import threading
import time
from pathlib import Path

import numpy as np
import structlog

from transcribe.base import BaseTranscriber, TranscriptionResult
from transcribe.factory import register

logger = structlog.get_logger(__name__)

_SAMPLE_RATE = 16000


@register("gigaam")
class GigaAMTranscriber(BaseTranscriber):
    # sherpa-onnx offline recognizer не возвращает пословные тайминги —
    # для диаризации эта ветка падает в fallback по majority vote.
    supports_word_timestamps = False

    def __init__(self, config: dict):
        self._config = config
        self._model_type = config.get("type", "ctc")
        self._recognizer = None
        self._lock = threading.Lock()

    def load(self) -> None:
        try:
            import sherpa_onnx
        except ImportError:
            raise ImportError("Установите sherpa-onnx: pip install sherpa-onnx")

        if self._model_type == "transducer":
            self._load_rnnt(sherpa_onnx)
        else:
            self._load_ctc(sherpa_onnx)

    def _load_ctc(self, sherpa_onnx) -> None:
        model_path = self._config.get("model_path", "models/gigaam-v3/model.int8.onnx")
        tokens_path = self._config.get("tokens_path", "models/gigaam-v3/tokens.txt")

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"GigaAM CTC модель не найдена: {model_path}\n"
                "Скачайте модель:\n"
                "  python download_models.py --sherpa"
            )

        logger.info("GigaAMTranscriber: загрузка CTC...", model=model_path)
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
            model=model_path,
            tokens=tokens_path,
            num_threads=2,
            sample_rate=_SAMPLE_RATE,
            feature_dim=80,
            decoding_method="greedy_search",
            debug=False,
        )
        logger.info("GigaAMTranscriber: CTC загружена")

    def _load_rnnt(self, sherpa_onnx) -> None:
        cfg = self._config
        encoder = cfg.get("encoder_path", "models/gigaam-v3-rnnt/encoder.int8.onnx")

        if not Path(encoder).exists():
            raise FileNotFoundError(
                f"GigaAM RNNT модель не найдена: {encoder}\n"
                "Скачайте модель:\n"
                "  python download_models.py --sherpa --rnnt"
            )

        logger.info("GigaAMTranscriber: загрузка RNNT...", encoder=encoder)
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_transducer(
            encoder=encoder,
            decoder=cfg.get("decoder_path", "models/gigaam-v3-rnnt/decoder.onnx"),
            joiner=cfg.get("joiner_path", "models/gigaam-v3-rnnt/joiner.int8.onnx"),
            tokens=cfg.get("rnnt_tokens_path", "models/gigaam-v3-rnnt/tokens.txt"),
            num_threads=2,
            sample_rate=_SAMPLE_RATE,
            feature_dim=80,
            decoding_method="greedy_search",
            debug=False,
        )
        logger.info("GigaAMTranscriber: RNNT загружена")

    def is_loaded(self) -> bool:
        return self._recognizer is not None

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        if not self.is_loaded():
            raise RuntimeError("GigaAMTranscriber: модель не загружена")

        audio_float = audio.astype(np.float32) / 32768.0

        t0 = time.monotonic()
        with self._lock:
            stream = self._recognizer.create_stream()
            stream.accept_waveform(_SAMPLE_RATE, audio_float)
            self._recognizer.decode_stream(stream)
            text = stream.result.text.strip() if stream.result.text else ""

        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.debug("GigaAMTranscriber: decoded", text=text[:60], latency_ms=round(elapsed_ms))

        return TranscriptionResult(text=text)
