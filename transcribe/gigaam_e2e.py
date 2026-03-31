"""GigaAM v3 e2e транскрибер (CTC/RNNT с пунктуацией и нормализацией)."""
import time

import numpy as np
import structlog
import torch

from transcribe.base import BaseTranscriber, TranscriptionResult

logger = structlog.get_logger(__name__)

_SAMPLE_RATE = 16000


class GigaAME2ETranscriber(BaseTranscriber):
    def __init__(self, config: dict):
        self._variant = config.get("variant", "e2e_ctc")
        self._model = None

    def load(self) -> None:
        from transformers import AutoModel

        revision = self._variant
        logger.info("GigaAME2E: загрузка модели...", variant=revision)
        wrapper = AutoModel.from_pretrained(
            "ai-sage/GigaAM-v3",
            revision=revision,
            trust_remote_code=True,
            low_cpu_mem_usage=False,
        )
        self._model = wrapper.model
        logger.info("GigaAME2E: модель загружена")

    def is_loaded(self) -> bool:
        return self._model is not None

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        if not self.is_loaded():
            raise RuntimeError("GigaAME2E: модель не загружена")

        audio_float = audio.astype(np.float32) / 32768.0
        wav = torch.from_numpy(audio_float).unsqueeze(0)
        length = torch.tensor([wav.shape[-1]])

        t0 = time.monotonic()
        with torch.inference_mode():
            encoded, encoded_len = self._model.forward(wav, length)
            text = self._model.decoding.decode(self._model.head, encoded, encoded_len)[0]

        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.debug("GigaAME2E: decoded", text=text[:60], latency_ms=round(elapsed_ms))
        return TranscriptionResult(text=text.strip())
