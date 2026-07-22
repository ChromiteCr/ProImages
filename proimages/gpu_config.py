def detect_device(override: str | None = None) -> str:
    """Pick the compute device.

    torch is imported lazily and only ships with the "heavy" extra, so a base
    install reports "cpu" -- accurate there, since torch is what provides GPU
    access and the modules that need one are unavailable in that install too.
    """
    if override is not None:
        return override

    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
