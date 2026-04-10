from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class WordTimestamp:
    text: str
    start: float  # секунды от начала сегмента
    end: float


@dataclass
class TranscriptionResult:
    text: str
    confidence: float = 0.0
    words: Optional[List[WordTimestamp]] = None


class BaseTranscriber(ABC):
    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def is_loaded(self) -> bool: ...

    @abstractmethod
    def transcribe(self, audio: np.ndarray) -> TranscriptionResult: ...
