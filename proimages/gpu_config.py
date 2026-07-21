import torch


def detect_device(override: str | None = None) -> str:
    if override is not None:
        return override
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
