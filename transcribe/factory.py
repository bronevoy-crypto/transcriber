"""Реестр транскриберов.

Новая модель подключается одним файлом в пакете transcribe/:
класс-наследник BaseTranscriber с декоратором @register("name").
Пакет автоимпортирует свои модули в __init__.py, поэтому
достаточно положить файл рядом — он сам зарегистрируется.
"""
from typing import Type

from transcribe.base import BaseTranscriber

_REGISTRY: dict[str, Type[BaseTranscriber]] = {}


def register(name: str):
    def _decorator(cls: Type[BaseTranscriber]) -> Type[BaseTranscriber]:
        if name in _REGISTRY:
            raise ValueError(f"Транскрибер '{name}' уже зарегистрирован")
        _REGISTRY[name] = cls
        return cls
    return _decorator


def available_models() -> list[str]:
    return sorted(_REGISTRY)


def create_transcriber(config: dict) -> BaseTranscriber:
    model_type = config.get("type", "whisper")
    if model_type not in _REGISTRY:
        raise ValueError(
            f"Неизвестная модель: {model_type}. "
            f"Доступно: {', '.join(available_models()) or '(реестр пуст)'}"
        )
    cls = _REGISTRY[model_type]
    transcriber = cls(config.get(model_type, {}))
    transcriber.load()
    return transcriber
