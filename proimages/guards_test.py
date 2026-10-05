import socket
import subprocess
import sys
import urllib.request
from pathlib import Path

import httpx
import pytest

from proimages import model_download

# Bound while this module is imported, during collection: what any module imported then would keep.
_GETADDRINFO_AT_IMPORT = socket.getaddrinfo

# Modules that only the optional extras ("heavy", "heif") or future stages bring in. The base install
# has none of them, so every module must import without them and import them lazily, inside functions.
OPTIONAL_MODULES = ("torch", "torchvision", "rawpy", "huggingface_hub", "transformers", "spandrel", "pi_heif")

_IMPORT_EVERYTHING = """
import importlib, sys
from pathlib import Path

for name in {blocked!r}:
    sys.modules[name] = None  # "import name" now raises ImportError, as on a base install

import proimages

# Walk the files: pkgutil.walk_packages silently skips directories that have no __init__.py.
root = Path(proimages.__file__).parent
names = []
for file in sorted(root.rglob("*.py")):
    if file.name.endswith("_test.py"):
        continue
    parts = file.relative_to(root.parent).with_suffix("").parts
    names.append(".".join(parts[:-1] if parts[-1] == "__init__" else parts))
for name in names:
    importlib.import_module(name)
print("\\n".join(names))
"""


def test_every_module_imports_without_the_optional_extras():
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_EVERYTHING.format(blocked=OPTIONAL_MODULES)],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    imported = set(result.stdout.split())
    assert {
        "proimages.api.app",
        "proimages.cli",
        "proimages.core.pipeline",
        "proimages.core.system.io",
        "proimages.core.denoise",  # a package is listed by its own name, not as pkg.__init__
    } <= imported


def test_model_downloads_are_refused():
    # Confirm conftest.py's guard is installed before calling, and use a repo that does not exist, so even a
    # run without the conftest (pytest --noconftest) cannot download real weights here.
    assert "proimages-test-models-" in str(model_download.MODELS_DIR)
    with pytest.raises(RuntimeError, match="must not download"):
        model_download.ensure_model("proimages-tests/never-downloaded")


def test_the_model_cache_is_an_empty_scratch_directory_not_the_users_cache():
    cache = model_download.MODELS_DIR

    assert cache.is_dir() and not any(cache.iterdir())
    assert cache.resolve() != (Path.home() / ".cache" / "proimages" / "models").resolve()


def test_network_connections_are_refused():
    with pytest.raises(RuntimeError, match="must not use the network"):
        socket.getaddrinfo("proimages.invalid", 443)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)  # a regressed guard must fail fast, not hang
        with pytest.raises(RuntimeError, match="must not use the network"):
            sock.connect(("192.0.2.1", 443))  # TEST-NET-1, never routed
        with pytest.raises(RuntimeError, match="must not use the network"):
            sock.connect_ex(("192.0.2.1", 443))
    # The clients that really get used, whatever proxy the environment configures for them.
    with pytest.raises(RuntimeError, match="must not use the network"):
        httpx.get("http://192.0.2.1/", timeout=2)
    with pytest.raises(RuntimeError, match="must not use the network"):
        urllib.request.urlopen("http://192.0.2.1/", timeout=2)


@pytest.mark.parametrize(
    "lookup, extra_arguments", [("getaddrinfo", (443,)), ("gethostbyname", ()), ("gethostbyname_ex", ())]
)
def test_name_lookups_are_refused(lookup, extra_arguments):
    with pytest.raises(RuntimeError, match="must not use the network"):
        getattr(socket, lookup)("proimages.invalid", *extra_arguments)


def test_the_network_guard_is_already_installed_during_collection():
    with pytest.raises(RuntimeError, match="must not use the network"):
        _GETADDRINFO_AT_IMPORT("proimages.invalid", 443)


def test_loopback_connections_stay_allowed():
    with socket.create_server(("127.0.0.1", 0)) as server:
        port = server.getsockname()[1]
        for host in ("127.0.0.1", "localhost", "LOCALHOST"):
            with socket.create_connection((host, port), timeout=1):
                pass


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="no Unix domain sockets")
def test_unix_socket_paths_stay_allowed(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # a relative name keeps the path under the ~100 byte limit on a socket path
    with socket.socket(socket.AF_UNIX) as server, socket.socket(socket.AF_UNIX) as client:
        server.bind("guard.sock")
        server.listen()
        client.connect("guard.sock")
