"""
Скачивание моделей GigaAM v3.

Запуск:
    python download_models.py          # CTC (рекомендуется для старта)
    python download_models.py --rnnt   # RNNT (точнее на ~1%, больше файлов)
    python download_models.py --all    # обе модели
"""
import argparse
from pathlib import Path


def download_ctc() -> None:
    from huggingface_hub import hf_hub_download

    print("Скачивание GigaAM v3 CTC...")
    for filename in ["model.int8.onnx", "tokens.txt"]:
        hf_hub_download(
            repo_id="csukuangfj/sherpa-onnx-nemo-ctc-giga-am-v3-russian-2025-12-16",
            filename=filename,
            local_dir="models/gigaam-v3",
        )
    print("GigaAM v3 CTC скачана: models/gigaam-v3/")


def download_rnnt() -> None:
    from huggingface_hub import hf_hub_download

    print("Скачивание GigaAM v3 RNNT...")
    for filename in ["encoder.int8.onnx", "decoder.onnx", "joiner.int8.onnx", "tokens.txt"]:
        hf_hub_download(
            repo_id="csukuangfj/sherpa-onnx-nemo-transducer-giga-am-v3-russian-2025-12-16",
            filename=filename,
            local_dir="models/gigaam-v3-rnnt",
        )
    print("GigaAM v3 RNNT скачана: models/gigaam-v3-rnnt/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rnnt", action="store_true", help="Скачать только RNNT")
    parser.add_argument("--all", action="store_true", help="Скачать CTC и RNNT")
    args = parser.parse_args()

    if args.all:
        download_ctc()
        download_rnnt()
    elif args.rnnt:
        download_rnnt()
    else:
        download_ctc()
