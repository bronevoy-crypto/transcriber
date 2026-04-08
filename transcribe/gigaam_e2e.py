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
        import shutil
        import tempfile
        import gigaam
        from pathlib import Path

        name = f"v3_{self._variant}"
        logger.info("GigaAME2E: загрузка модели...", variant=name)
        models_dir = Path(__file__).parent.parent / "models" / "gigaam"
        models_dir.mkdir(parents=True, exist_ok=True)

        # sentencepiece (C++) не открывает файлы с кириллицей в пути на Windows.
        # Копируем модели во временную папку с ASCII-путём перед загрузкой.
        try:
            str(models_dir).encode("ascii")
            load_dir = models_dir
        except UnicodeEncodeError:
            load_dir = Path(tempfile.gettempdir()) / "gigaam_models"
            load_dir.mkdir(exist_ok=True)
            for f in models_dir.iterdir():
                shutil.copy2(f, load_dir / f.name)

        self._model = gigaam.load_model(name, download_root=str(load_dir))
        logger.info("GigaAME2E: модель загружена")

    def is_loaded(self) -> bool:
        return self._model is not None

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        if not self.is_loaded():
            raise RuntimeError("GigaAME2E: модель не загружена")

        audio_float = audio.astype(np.float32) / 32768.0
        rms = np.sqrt(np.mean(audio_float ** 2))
        if rms > 0.001:
            audio_float = np.clip(audio_float * (0.05 / rms), -1.0, 1.0)
        wav = torch.from_numpy(audio_float).to(self._model._device).to(self._model._dtype).unsqueeze(0)
        length = torch.tensor([wav.shape[-1]], device=self._model._device)

        t0 = time.monotonic()
        with torch.inference_mode():
            encoded, encoded_len = self._model.forward(wav, length)
            text, _ = self._model._decode(encoded, encoded_len, int(length[0].item()), False)

        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.debug("GigaAME2E: decoded", text=text[:60], latency_ms=round(elapsed_ms))
        return TranscriptionResult(text=text.strip())
