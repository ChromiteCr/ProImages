"""Suite-wide guards: no test may download model weights or reach the network."""

import ipaddress
import shutil
import socket
import tempfile
from pathlib import Path

import pytest

_LOCAL_HOSTNAMES = {"", "localhost"}
_PROXY_VARIABLES = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")

# Installed in pytest_configure, before collection, so import-time code and session-scoped fixtures are
# guarded too. A test's own function-scoped monkeypatch (a fake model loader, say) restores these guarded values.
_patches = pytest.MonkeyPatch()
_models_dir: Path | None = None


def _is_local(host) -> bool:
    if isinstance(host, bytes):
        host = host.decode(errors="replace")
    if host is None or host.lower() in _LOCAL_HOSTNAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _refuse_download(*args, **kwargs):
    raise RuntimeError("tests must not download model weights; monkeypatch the loader instead")


def _local_only(real):
    def guarded(self, address):
        # AF_UNIX addresses are paths, not (host, port) tuples.
        if isinstance(address, tuple) and not _is_local(address[0]):
            raise RuntimeError(f"tests must not use the network (tried to connect to {address[0]})")
        return real(self, address)

    return guarded


def _local_hosts_only(real):
    def guarded(host, *args, **kwargs):
        if not _is_local(host):
            raise RuntimeError(f"tests must not use the network (tried to resolve {host})")
        return real(host, *args, **kwargs)

    return guarded


def pytest_configure(config):
    global _models_dir

    # Clients would reach the internet through a proxy on loopback, which the socket guards allow. "*" bypasses
    # every proxy, and also stops urllib and requests from falling back to the macOS system proxy.
    for name in _PROXY_VARIABLES:
        _patches.delenv(name, raising=False)
        _patches.delenv(name.lower(), raising=False)
    _patches.setenv("NO_PROXY", "*")
    _patches.setenv("no_proxy", "*")

    # An early-bound reference to the real ensure_model then misses the cache and meets the socket guards.
    _models_dir = Path(tempfile.mkdtemp(prefix="proimages-test-models-"))
    _patches.setattr("proimages.model_download.ensure_model", _refuse_download)
    _patches.setattr("proimages.model_download.MODELS_DIR", _models_dir)

    _patches.setattr(socket.socket, "connect", _local_only(socket.socket.connect))
    _patches.setattr(socket.socket, "connect_ex", _local_only(socket.socket.connect_ex))
    for name in ("getaddrinfo", "gethostbyname", "gethostbyname_ex"):
        _patches.setattr(socket, name, _local_hosts_only(getattr(socket, name)))


def pytest_unconfigure(config):
    _patches.undo()
    if _models_dir is not None:
        shutil.rmtree(_models_dir, ignore_errors=True)
