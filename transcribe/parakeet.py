"""NVIDIA Parakeet TDT v3 (мультиязычный) через sherpa-onnx."""
import threading
import time
from pathlib import Path

import numpy as np
import structlog

from transcribe.base import BaseTranscriber, TranscriptionResult, WordTimestamp
from transcribe.factory import register

logger = structlog.get_logger(__name__)

_SAMPLE_RATE = 16000


def _tokens_to_words(tokens: list[str], timestamps: list[float]) -> list[WordTimestamp]:
    # Parakeet sherpa-onnx отдаёт BPE-сабворды с ведущим пробелом как маркером
    # начала нового слова. Склеиваем подряд идущие сабворды до следующего пробела.
    words: list[WordTimestamp] = []
    cur_text = ""
    cur_start = 0.0
    cur_end = 0.0

    def _flush():
        if cur_text.strip():
            words.append(WordTimestamp(text=cur_text.strip(), start=cur_start, end=cur_end))

    for tok, ts in zip(tokens, timestamps):
        is_word_start = tok.startswith(" ")
        if is_word_start:
            _flush()
            cur_text = tok.lstrip()
            cur_start = ts
            cur_end = ts
        else:
            cur_text += tok
            cur_end = ts
    _flush()
    return words


@register("parakeet")
class ParakeetTranscriber(BaseTranscriber):
    supports_word_timestamps = True

    def __init__(self, config: dict):
        self._encoder = config.get("encoder_path", "models/parakeet/encoder.int8.onnx")
        self._decoder = config.get("decoder_path", "models/parakeet/decoder.int8.onnx")
        self._joiner = config.get("joiner_path", "models/parakeet/joiner.int8.onnx")
        self._tokens = config.get("tokens_path", "models/parakeet/tokens.txt")
        self._num_threads = int(config.get("num_threads", 2))
        self._recognizer = None
        self._lock = threading.Lock()

    def load(self) -> None:
        try:
            import sherpa_onnx
        except ImportError:
            raise ImportError("Установите sherpa-onnx: pip install sherpa-onnx")

        if not Path(self._encoder).exists():
            print("Parakeet: модель не найдена, скачиваю (~640 МБ)...")
            from download_models import download_parakeet
            download_parakeet()

        logger.info("ParakeetTranscriber: загрузка модели...", encoder=self._encoder)
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=self._encoder,
            decoder=self._decoder,
            joiner=self._joiner,
            tokens=self._tokens,
            model_type="nemo_transducer",
            num_threads=self._num_threads,
            sample_rate=_SAMPLE_RATE,
            feature_dim=128,
            decoding_method="greedy_search",
            debug=False,
        )
        logger.info("ParakeetTranscriber: модель загружена")

    def is_loaded(self) -> bool:
        return self._recognizer is not None

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        if not self.is_loaded():
            raise RuntimeError("ParakeetTranscriber: модель не загружена")

        audio_float = audio.astype(np.float32) / 32768.0

        t0 = time.monotonic()
        with self._lock:
            stream = self._recognizer.create_stream()
            stream.accept_waveform(_SAMPLE_RATE, audio_float)
            self._recognizer.decode_stream(stream)
            result = stream.result

        text = (result.text or "").strip()
        words = _tokens_to_words(list(result.tokens), list(result.timestamps)) if result.tokens else None

        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.debug("ParakeetTranscriber: decoded", text=text[:60], latency_ms=round(elapsed_ms))

        return TranscriptionResult(text=text, words=words or None)
