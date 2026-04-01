"""Silero VAD — детекция речевой активности."""
import numpy as np
import structlog
import torch

logger = structlog.get_logger(__name__)

_SAMPLE_RATE = 16000


class VADProcessor:
    def __init__(self, threshold: float = 0.5, min_speech_ms: int = 250):
        self._threshold = threshold
        self._min_speech_samples = int(_SAMPLE_RATE * min_speech_ms / 1000)
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        logger.info("VADProcessor: загрузка silero VAD...")
        self._model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=False,
        )
        self._model.eval()
        logger.info("VADProcessor: модель загружена")

    def is_speech(self, audio: np.ndarray) -> bool:
        if len(audio) < self._min_speech_samples:
            return False

        frame_size = 512
        speech_frames = 0
        total_frames = 0

        for i in range(0, len(audio) - frame_size + 1, frame_size):
            frame = audio[i:i + frame_size]
            audio_float = torch.from_numpy(frame.astype(np.float32) / 32768.0)
            with torch.no_grad():
                prob = self._model(audio_float, _SAMPLE_RATE).item()
            if prob >= self._threshold:
                speech_frames += 1
            total_frames += 1

        return total_frames > 0 and (speech_frames / total_frames) >= self._threshold
