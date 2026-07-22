import builtins

from proimages.gpu_config import detect_device


def test_detect_device_returns_supported_backend():
    assert detect_device() in {"cuda", "mps", "cpu"}


def test_detect_device_honors_override():
    assert detect_device(override="cpu") == "cpu"


def test_detect_device_falls_back_to_cpu_without_torch(monkeypatch):
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("No module named 'torch'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    assert detect_device() == "cpu"
