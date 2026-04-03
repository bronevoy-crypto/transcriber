"""Проверка окружения — запусти на ноуте и скинь вывод."""
import sys
print("Python:", sys.version)

try:
    import torch
    print("torch:", torch.__version__)
    print("CUDA:", torch.cuda.is_available())
except Exception as e:
    print("torch ERROR:", e)

try:
    import pyannote.audio
    print("pyannote.audio:", pyannote.audio.__version__)
except Exception as e:
    print("pyannote.audio ERROR:", e)

try:
    import speechbrain
    print("speechbrain:", speechbrain.__version__)
except Exception as e:
    print("speechbrain ERROR:", e)

try:
    import numpy
    print("numpy:", numpy.__version__)
except Exception as e:
    print("numpy ERROR:", e)
