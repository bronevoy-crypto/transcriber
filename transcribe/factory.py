from transcribe.base import BaseTranscriber


def create_transcriber(config: dict) -> BaseTranscriber:
    model_type = config.get("type", "whisper")

    if model_type == "gigaam":
        from transcribe.gigaam import GigaAMTranscriber
        transcriber = GigaAMTranscriber(config.get("gigaam", {}))
    elif model_type == "whisper":
        from transcribe.whisper import WhisperTranscriber
        transcriber = WhisperTranscriber(config.get("whisper", {}))
    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}. Доступные: gigaam, whisper")

    transcriber.load()
    return transcriber
