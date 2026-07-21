from proimages.gpu_config import detect_device


def test_detect_device_returns_supported_backend():
    assert detect_device() in {"cuda", "mps", "cpu"}


def test_detect_device_honors_override():
    assert detect_device(override="cpu") == "cpu"
