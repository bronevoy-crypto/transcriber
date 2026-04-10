"""Запись транскрипции в JSON."""
import json
import threading
from datetime import datetime
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


class JSONWriter:
    def __init__(self, output_dir: str = "meetings"):
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(exist_ok=True)
        self._filepath: Path | None = None
        self._segments: list[dict] = []
        self._lock = threading.Lock()
        self._meeting_start: datetime | None = None
        self._config: dict | None = None

    def start_meeting(self, config: dict | None = None) -> None:
        self._meeting_start = datetime.now()
        self._config = config
        filename = self._meeting_start.strftime("%Y-%m-%d_%H-%M-%S") + ".json"
        self._filepath = self._output_dir / filename
        self._segments = []
        logger.info("JSONWriter: файл встречи создан", path=str(self._filepath))

    def write_segment(self, start: float, end: float, text: str, speaker: str = "SPEAKER_00", words: list | None = None) -> None:
        segment = {
            "start": round(start, 2),
            "end": round(end, 2),
            "speaker": speaker,
            "text": text,
        }
        if words:
            segment["words"] = [{"text": w.text, "start": w.start, "end": w.end} for w in words]
        with self._lock:
            self._segments.append(segment)
            should_flush = len(self._segments) % 3 == 0
        if should_flush:
            self._flush()

    def finish(self) -> str | None:
        self._flush()
        if self._filepath:
            logger.info("JSONWriter: встреча завершена", path=str(self._filepath), segments=len(self._segments))
            return str(self._filepath)
        return None

    def _flush(self) -> None:
        if not self._filepath:
            return
        with self._lock:
            segments_snapshot = list(self._segments)
        data = {
            "meeting_date": self._meeting_start.isoformat() if self._meeting_start else None,
            "config": self._config,
            "segments": segments_snapshot,
        }
        self._filepath.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
