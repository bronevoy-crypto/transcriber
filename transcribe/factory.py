from transcribe.base import BaseTranscriber


def create_transcriber(config: dict) -> BaseTranscriber:
    model_type = config.get("type", "whisper")

    if model_type == "gigaam_e2e":
        from transcribe.gigaam_e2e import GigaAME2ETranscriber
        transcriber = GigaAME2ETranscriber(config.get("gigaam_e2e", {}))
    elif model_type == "gigaam":
        from transcribe.gigaam import GigaAMTranscriber
        transcriber = GigaAMTranscriber(config.get("gigaam", {}))
    elif model_type == "whisper":
        from transcribe.whisper import WhisperTranscriber
        transcriber = WhisperTranscriber(config.get("whisper", {}))
    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}. Доступные: gigaam_e2e, gigaam, whisper")

    transcriber.load()
    return transcriber
