"""Диаризация говорящих через pyannote-audio."""
import numpy as np
import structlog
import torch

logger = structlog.get_logger(__name__)


def _fix_torch_load_compat():
    """PyTorch 2.6+ defaults to weights_only=True, breaking old pyannote checkpoints.

    Patches lightning_fabric's _load to use weights_only=False for trusted sources.
    """
    try:
        import lightning_fabric.utilities.cloud_io as _cloud_io
        import torch
        _orig_load = _cloud_io._load

        def _patched_load(path_or_url, map_location=None, **kwargs):
            kwargs["weights_only"] = False  # force: pyannote checkpoints need full unpickling
            return _orig_load(path_or_url, map_location=map_location, **kwargs)

        _cloud_io._load = _patched_load
    except Exception:
        pass


def _fix_pyannote_compat():
    """Monkey-patch: pyannote 3.3.x + huggingface_hub 0.24+ используют разные имена параметра.

    pyannote 3.3.x передаёт use_auth_token в hf_hub_download,
    но новый huggingface_hub ждёт token.
    """
    try:
        import huggingface_hub
        orig_download = huggingface_hub.hf_hub_download
        def _patched_download(*args, use_auth_token=None, token=None, **kwargs):
            # normalize: use_auth_token → token
            if use_auth_token is not None and token is None:
                token = use_auth_token
            return orig_download(*args, token=token, **kwargs)
        huggingface_hub.hf_hub_download = _patched_download
        # Also patch the import that pyannote already did
        import pyannote.audio.core.pipeline as _pp
        _pp.hf_hub_download = _patched_download
    except Exception:
        pass


def _fix_torchaudio_compat():
    """Monkey-patch для совместимости torchaudio 2.x с pyannote 3.x."""
    try:
        import torchaudio

        if not hasattr(torchaudio, "AudioMetaData"):
            from dataclasses import dataclass
            @dataclass
            class AudioMetaData:
                sample_rate: int
                num_frames: int
                num_channels: int
                bits_per_sample: int
                encoding: str
            torchaudio.AudioMetaData = AudioMetaData

        if not hasattr(torchaudio, "info"):
            import soundfile as sf
            def _info(path, **kwargs):
                info = sf.info(str(path))
                return torchaudio.AudioMetaData(
                    sample_rate=info.samplerate,
                    num_frames=info.frames,
                    num_channels=info.channels,
                    bits_per_sample=16,
                    encoding="PCM_S",
                )
            torchaudio.info = _info

        if not hasattr(torchaudio, "list_audio_backends"):
            torchaudio.list_audio_backends = lambda: ["soundfile"]

        if not hasattr(torchaudio, "get_audio_backend"):
            torchaudio.get_audio_backend = lambda: "soundfile"

        if not hasattr(torchaudio, "set_audio_backend"):
            torchaudio.set_audio_backend = lambda backend: None

        if not hasattr(torchaudio, "load"):
            import soundfile as sf
            import numpy as np
            def _load(path, **kwargs):
                data, sr = sf.read(str(path), dtype="float32", always_2d=True)
                return torch.from_numpy(data.T), sr
            torchaudio.load = _load

    except Exception:
        pass


class Diarizer:
    def __init__(self, hf_token: str, device: str = "cpu"):
        self._hf_token = hf_token
        # Fallback to CPU if CUDA not available in this torch build
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("Diarizer: CUDA недоступна, используем CPU")
            device = "cpu"
        self._device = torch.device(device)
        self._pipeline = None
        self._speaker_map: dict[str, str] = {}  # pyannote_id -> наш SPEAKER_XX
        self._next_id = 0

    def load(self) -> None:
        _fix_torch_load_compat()
        _fix_torchaudio_compat()
        _fix_pyannote_compat()
        from pyannote.audio import Pipeline
        logger.info("Diarizer: загрузка модели...")
        self._pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=self._hf_token,
        )
        self._pipeline.to(self._device)
        logger.info("Diarizer: модель загружена")

    def build_timeline(self, audio: np.ndarray, sample_rate: int = 16000) -> list[dict]:
        """Диаризировать ПОЛНОЕ аудио и вернуть таймлайн со стабильными ID дикторов.

        Правильный подход: запускается один раз на всё аудио,
        а не на каждый сегмент отдельно.

        Возвращает список dict{start, end, speaker} с SPEAKER_00/01/...
        """
        if self._pipeline is None:
            return []

        logger.info("Diarizer: диаризация полного аудио", duration_s=round(len(audio) / sample_rate, 1))
        waveform = torch.from_numpy(audio.astype(np.float32) / 32768.0).unsqueeze(0).to(self._device)
        try:
            diarization = self._pipeline({"waveform": waveform, "sample_rate": sample_rate})
        except Exception as e:
            logger.warning("Diarizer: ошибка диаризации", error=str(e))
            return []

        timeline = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if speaker not in self._speaker_map:
                self._speaker_map[speaker] = f"SPEAKER_{self._next_id:02d}"
                self._next_id += 1
                logger.info("Diarizer: новый спикер", id=self._speaker_map[speaker])
            timeline.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": self._speaker_map[speaker],
            })

        unique = len(set(t["speaker"] for t in timeline))
        logger.info("Diarizer: таймлайн готов", intervals=len(timeline), speakers=unique)
        return timeline

    def speaker_at(self, timeline: list[dict], start: float, end: float) -> str:
        """Найти доминирующего диктора в заданном временном интервале по таймлайну."""
        speaker_times: dict[str, float] = {}
        for item in timeline:
            overlap_start = max(item["start"], start)
            overlap_end = min(item["end"], end)
            if overlap_end > overlap_start:
                dur = overlap_end - overlap_start
                speaker_times[item["speaker"]] = speaker_times.get(item["speaker"], 0) + dur
        if not speaker_times:
            return "SPEAKER_00"
        return max(speaker_times, key=speaker_times.get)

    def diarize(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """[УСТАРЕЛО] Определить доминирующего говорящего в одном сегменте.

        Не рекомендуется: работает некорректно для разных сегментов одной встречи.
        Используйте build_timeline() + speaker_at() вместо этого.
        """
        if self._pipeline is None:
            return "SPEAKER_00"

        waveform = torch.from_numpy(audio.astype(np.float32) / 32768.0).unsqueeze(0).to(self._device)
        try:
            diarization = self._pipeline({"waveform": waveform, "sample_rate": sample_rate})
        except Exception as e:
            logger.warning("Diarizer: ошибка диаризации", error=str(e))
            return "SPEAKER_00"

        speaker_times: dict[str, float] = {}
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            duration = turn.end - turn.start
            speaker_times[speaker] = speaker_times.get(speaker, 0) + duration

        if not speaker_times:
            return "SPEAKER_00"

        dominant = max(speaker_times, key=speaker_times.get)
        if dominant not in self._speaker_map:
            self._speaker_map[dominant] = f"SPEAKER_{self._next_id:02d}"
            self._next_id += 1
            logger.info("Diarizer: новый спикер", id=self._speaker_map[dominant])

        return self._speaker_map[dominant]
