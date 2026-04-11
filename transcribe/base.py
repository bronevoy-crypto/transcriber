from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class WordTimestamp:
    text: str
    start: float
    end: float


@dataclass
class TranscriptionResult:
    text: str
    confidence: float = 0.0
    words: Optional[List[WordTimestamp]] = None


class BaseTranscriber(ABC):
    # Если модель отдаёт word-level timestamps — переопределить в подклассе.
    # От этого флага зависит способ привязки спикеров в диаризации.
    supports_word_timestamps: bool = False

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def is_loaded(self) -> bool: ...

    @abstractmethod
    def transcribe(self, audio: np.ndarray) -> TranscriptionResult: ...
