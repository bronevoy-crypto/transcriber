# Meeting Transcriber

Локальный транскрибер совещаний для Windows. Пишет системный звук + микрофон, режет по паузам, транскрибирует, определяет кто говорит. Результат — JSON с таймкодами и метками дикторов.

Всё работает на локальной машине, никуда не отправляется.

---

## Содержание

- [Быстрый старт](#быстрый-старт)
- [Требования](#требования)
- [Установка](#установка)
- [Настройка .env и диаризации](#настройка-env-и-диаризации)
- [Запуск](#запуск)
- [Формат вывода](#формат-вывода)
- [Модели](#модели)
- [Архитектура](#архитектура)
- [Как добавить свою модель](#как-добавить-свою-модель)
- [Troubleshooting](#troubleshooting)

---

## Быстрый старт

```bash
git clone https://github.com/HRYNdev/transcriber.git
cd transcriber
pip install -r requirements.txt
pip install --no-deps git+https://github.com/salute-developers/GigaAM.git
copy config.example.yaml config.yaml
copy .env.example .env
# вписать HF_TOKEN в .env
python main.py
```

Говори, затем `Ctrl+C` → в `meetings/` появится JSON.

---

## Требования

- Windows 10/11
- Python 3.10+
- NVIDIA GPU — желательно, но не обязательно (GigaAM работает и на CPU)
- Аккаунт на HuggingFace — нужен для диаризации
- ~2 ГБ на диске под модели (GigaAM + pyannote)

---

## Установка

```bash
git clone https://github.com/HRYNdev/transcriber.git
cd transcriber
pip install -r requirements.txt

# GigaAM ставится отдельно из git (не в PyPI)
pip install --no-deps git+https://github.com/salute-developers/GigaAM.git

# Скопировать примеры конфигов
copy config.example.yaml config.yaml
copy .env.example .env
```

Скачать модели GigaAM (одноразово):

```bash
python download_models.py
```

---

## Настройка .env и диаризации

Диаризация (определение "кто говорит") работает через `pyannote-audio` и требует токен HuggingFace.

1. Зарегистрироваться на [huggingface.co](https://huggingface.co).
2. Принять условия использования:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
3. Создать токен: https://huggingface.co/settings/tokens → **New token** → **Read**.
4. Вставить токен в `.env`:

```env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
```

Файл `.env` не коммитится в репозиторий (см. `.gitignore`). Для локальной разработки достаточно одного файла, для деплоя — переменная окружения того же имени.

Если диаризация не нужна — в `config.yaml`:

```yaml
diarization:
  enabled: false
```

---

## Запуск

```bash
python main.py
```

Запускать **до** митинга. Остановить — `Ctrl+C`. После остановки автоматически запустится диаризация всей записи (в отдельном процессе, см. [Архитектура](#архитектура)).

Результат — в папке `meetings/` (имя файла = дата+время старта).

---

## Формат вывода

`meetings/2026-04-11_17-34-20.json`:

```json
{
  "meeting_date": "2026-04-11T17:34:20",
  "config": {
    "model": "gigaam_e2e",
    "sample_rate": 16000,
    "chunk_ms": 500,
    "silence_duration": 0.2,
    "diarization": true
  },
  "segments": [
    {
      "start": 2.57,
      "end": 10.07,
      "speaker": "SPEAKER_00",
      "text": "Слышь, ребята, опять наша ERP тормозит."
    },
    {
      "start": 10.49,
      "end": 25.93,
      "speaker": "SPEAKER_01",
      "text": "Классическая ситуация..."
    }
  ]
}
```

`SPEAKER_00`, `SPEAKER_01`, ... — стабильные идентификаторы в рамках одной встречи. Для перевода в настоящие имена нужно дополнительное сопоставление (эта логика — вне транскрибера).

---

## Модели

Модель выбирается в `config.yaml`:

```yaml
model:
  type: "gigaam_e2e"   # gigaam_e2e | gigaam | whisper | parakeet | tone
```

| Модель | Язык | Word-level TS | WER на клиентском аудио\* | Комментарий |
|--------|------|:-------------:|:-------------------------:|-------------|
| `gigaam_e2e` | ru | ✓ | **6.8%** | Рекомендуется. Числа цифрами, пунктуация, лучшее качество для русского. |
| `gigaam` | ru | — | ≈8–10% | GigaAM v3 через sherpa-onnx (CTC/RNNT), без слов, чуть быстрее. |
| `whisper` | multi | ✓ | 10–15% | Faster-Whisper large-v3. Нужна GPU с 8+ ГБ. Мультиязычный. |
| `parakeet` | multi | ✓† | 25.7% | NVIDIA Parakeet TDT v3 через sherpa-onnx. 25 языков включая RU. |
| `tone` | ru | — | 27.4% | T-one от T-Tech. Заточен под 8 kHz телефонию. Без пунктуации. |

\* WER померен на одной 103-секундной записи (`tts_tests/client_test2.wav`). На реальных данных цифры будут отличаться — используйте как ориентир.

† `parakeet` отдаёт только token-level тайминги, пословные восстанавливаются из BPE-сабвордов — это приближение, на границах слов возможны отклонения ±50 мс.

**Про word-level timestamps**: если модель их не даёт, диаризация работает в режиме "один доминирующий спикер на весь сегмент" (majority vote). С word-level можно переключать спикера на каждом слове — точнее на границах реплик.

### Конфиг моделей

```yaml
model:
  type: "gigaam_e2e"
  gigaam_e2e:
    variant: "e2e_ctc"       # e2e_ctc или e2e_rnnt
  gigaam:
    type: "ctc"              # ctc или transducer
    model_path: "models/gigaam-v3/model.int8.onnx"
    tokens_path: "models/gigaam-v3/tokens.txt"
  whisper:
    model: "large-v3"
    language: "ru"
    device: "cuda"
    compute_type: "float16"
  parakeet:
    encoder_path: "models/parakeet/encoder.int8.onnx"
    decoder_path: "models/parakeet/decoder.int8.onnx"
    joiner_path: "models/parakeet/joiner.int8.onnx"
    tokens_path: "models/parakeet/tokens.txt"
  tone:
    decoder: "greedy"        # или "beam_search" (требует kenlm)
```

Скачать Parakeet (640 МБ, один раз):

```bash
mkdir models/parakeet && cd models/parakeet
curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2
tar -xf sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2
mv sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/* .
rm -rf sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8 sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2
```

T-one качает модель сам с HuggingFace при первом запуске (~300 МБ в `~/.cache/huggingface`).

---

## Архитектура

### Пайплайн обработки

```
┌───────────────────┐
│  AudioCapture     │   WASAPI loopback + микрофон
│  (audio/capture)  │   (два потока PortAudio, микс 50/50)
└─────────┬─────────┘
          │  chunk_ms × int16 моно 16kHz
          ▼
┌───────────────────┐
│   VADProcessor    │   Silero VAD — is_speech(chunk) → bool
│   (audio/vad)     │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  State machine    │   speech_buffer / pending_silence / silence_start
│  в main.py        │   Режет на сегменты когда пауза ≥ silence_duration
└─────────┬─────────┘
          │  audio_segment (np.int16)
          ▼
┌───────────────────┐
│   Transcriber     │   BaseTranscriber → TranscriptionResult(text, words)
│  (transcribe/*)   │   Выбор модели через реестр в factory.py
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│    JSONWriter     │   Инкрементально пишет сегменты в meetings/*.json
│  (output/writer)  │
└─────────┬─────────┘
          │
          │   ─── Ctrl+C → запись закончилась ───
          │
          ▼
┌───────────────────┐
│  Diarizer         │   Запускает pyannote в отдельном subprocess
│ (transcribe/      │   (изоляция тяжёлого torch+pyannote от основного
│  diarizer.py →    │   процесса, гарантированное освобождение памяти)
│  diarize_worker)  │
└─────────┬─────────┘
          │  timeline = [{start, end, speaker}]
          ▼
┌───────────────────┐
│  assign_speakers  │   Пришивает спикеров к сегментам:
│  _by_word   (или  │   — есть word-level → по каждому слову
│   speaker_at)     │   — нет → доминирующий спикер на сегмент
└─────────┬─────────┘
          │
          ▼
    финальный JSON
    в meetings/
```

### Несколько заметок по дизайну

Диаризация идёт в отдельном subprocess — pyannote тащит за собой половину питона (torch, speechbrain, k2) и на py3.13 это всё любит падать при загрузке. Плюс GPU-память оно не всегда отпускает. Проще прибить процесс когда закончили, чем чинить их утечки. Сам вызов и таймаут — в [`transcribe/diarizer.py`](transcribe/diarizer.py), сам pyannote — в [`transcribe/diarize_worker.py`](transcribe/diarize_worker.py).

Loopback и микрофон мержатся **до** VAD, не после. Иначе пришлось бы гонять VAD на двух потоках, сшивать их по времени и разбираться кто говорил первым. Проще смешать чанки в один поток — а диаризация в конце всё равно разложит по спикерам.

В [`audio/capture.py`](audio/capture.py) буферы — обычные list'ы, а не `queue.Queue`. PortAudio-колбэки дёргают `append`, а он под GIL атомарен. `Queue.put` при переполнении встаёт на блокировку, а залочить C-колбэк PortAudio — гарантированный deadlock всего пайплайна.

В `main.py` есть насильственная нарезка монолога каждые 180 секунд (`max_segment_chunks`). Реплики без пауз дольше трёх минут подвешивают VAD и раздувают буфер — проще резать принудительно, потом диаризация всё равно соберёт обратно.

Модели подключаются через реестр с декоратором `@register("name")`, смотри `transcribe/factory.py`. Чтобы добавить новую — кладёшь файл в `transcribe/`, пакет сам его подхватит при импорте (см. `transcribe/__init__.py`). Подробности и пример в разделе [Как добавить свою модель](#как-добавить-свою-модель).

### Структура каталогов

```
transcriber/
├── main.py                  # точка входа, машина состояний цикла записи
├── config.yaml              # рабочий конфиг (не коммитится)
├── config.example.yaml      # пример конфига (коммитится)
├── .env                     # токены (не коммитится)
├── .env.example             # пример .env (коммитится)
├── requirements.txt
├── download_models.py       # скачивание GigaAM моделей
│
├── audio/
│   ├── capture.py           # WASAPI loopback + микрофон
│   └── vad.py               # Silero VAD
│
├── transcribe/
│   ├── __init__.py          # автоимпорт модулей (запускает @register)
│   ├── base.py              # BaseTranscriber, TranscriptionResult, WordTimestamp
│   ├── factory.py           # реестр: @register, create_transcriber, available_models
│   ├── gigaam_e2e.py        # @register("gigaam_e2e") — рекомендуемый
│   ├── gigaam.py            # @register("gigaam")
│   ├── whisper.py           # @register("whisper")
│   ├── parakeet.py          # @register("parakeet")
│   ├── tone.py              # @register("tone")
│   ├── diarizer.py          # фасад, запускает diarize_worker
│   └── diarize_worker.py    # pyannote в отдельном процессе
│
├── output/
│   └── writer.py            # JSONWriter (инкрементальная запись)
│
├── models/                  # не коммитится
│   ├── gigaam-v3/
│   ├── parakeet/
│   └── pyannote/
│
└── meetings/                # результаты (не коммитятся)
```

---

## Как добавить свою модель

Допустим, появилась новая модель `superasr`. Всё что нужно:

1. Создать файл [`transcribe/superasr.py`](transcribe/):

```python
"""SuperASR транскрибер."""
import numpy as np
import structlog

from transcribe.base import BaseTranscriber, TranscriptionResult, WordTimestamp
from transcribe.factory import register

logger = structlog.get_logger(__name__)


@register("superasr")
class SuperAsrTranscriber(BaseTranscriber):
    # True если модель отдаёт пословные тайминги, False — только текст
    supports_word_timestamps = True

    def __init__(self, config: dict):
        self._model_path = config.get("model_path", "models/superasr/model.bin")
        self._model = None

    def load(self) -> None:
        import superasr
        self._model = superasr.load(self._model_path)
        logger.info("SuperASR: модель загружена")

    def is_loaded(self) -> bool:
        return self._model is not None

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        # audio — np.int16 16kHz моно
        audio_f32 = audio.astype(np.float32) / 32768.0
        result = self._model.decode(audio_f32)

        words = [
            WordTimestamp(text=w.text, start=w.start, end=w.end)
            for w in result.words
        ] if result.words else None

        return TranscriptionResult(text=result.text, words=words)
```

2. Добавить секцию в `config.yaml`:

```yaml
model:
  type: "superasr"
  superasr:
    model_path: "models/superasr/model.bin"
```

3. Запустить `python main.py`. Всё.

Ядро (`factory.py`, `main.py`) править не нужно. Реестр соберёт транскрибер автоматически. Если зависимости модели не установлены — пакет логирует debug и продолжает работать с остальными моделями.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'gigaam'`**
GigaAM не ставится из PyPI — нужно отдельно: `pip install --no-deps git+https://github.com/salute-developers/GigaAM.git`.

**`FileNotFoundError: models/gigaam-v3/model.int8.onnx`**
Модели не скачаны. Запустить `python download_models.py`.

**`hf_token не задан`**
Не создан `.env` или токен без префикса `hf_`. Скопировать `.env.example` → `.env` и вставить реальный токен.

**Диаризация падает с `AttributeError: module 'torchaudio' has no attribute 'info'`**
Конфликт torchaudio 2.x и pyannote 3.x. Патчи совместимости в [`transcribe/diarizer.py`](transcribe/diarizer.py) должны это закрывать. Если не помогло — проверить что в subprocess-воркере загружается нужная версия.

**`OutOfMemoryError` на Whisper**
Поставить `compute_type: "int8"` в секции whisper, либо переключить на `gigaam_e2e` (работает и на CPU).

**Кириллица ломается в консоли Windows**
`chcp 65001` перед запуском, либо запускать через Windows Terminal — там UTF-8 из коробки.

**Диаризация отработала, но все сегменты `SPEAKER_00`**
Запись слишком короткая или один голос. Диаризация требует ≥1 сек аудио с разными голосами. Также проверить что в конфиге `diarization.enabled: true`.

**Реплика длиной несколько минут без точек**
VAD не увидел паузы. Понизить `silence_duration` в `config.yaml` или увеличить `vad.threshold`. Либо включить принудительную нарезку (она уже есть в коде, 180 сек — см. `max_segment_chunks` в `main.py`).
