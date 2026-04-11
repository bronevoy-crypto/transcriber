"""Пакет transcribe.

При импорте пакета автоматически подгружает все модули-транскриберы —
декораторы @register срабатывают и заполняют реестр в factory.py.
Файлы, не являющиеся транскриберами, перечислены в _SKIP.
"""
import importlib
import pkgutil
import structlog
from pathlib import Path

_logger = structlog.get_logger(__name__)

_SKIP = {"base", "factory", "diarizer", "diarize_worker"}

_here = Path(__file__).parent
for _finder, _name, _ispkg in pkgutil.iter_modules([str(_here)]):
    if _name.startswith("_") or _name in _SKIP:
        continue
    try:
        importlib.import_module(f"{__name__}.{_name}")
    except Exception as e:
        # Не падаем если у пользователя не установлены зависимости
        # конкретной модели (faster-whisper, sherpa-onnx и т.п.).
        _logger.debug("transcribe: модуль не загружен", module=_name, error=str(e))
