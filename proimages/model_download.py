import os
from pathlib import Path

MODELS_DIR = Path(os.environ.get("PROIMAGES_MODELS_DIR", Path.home() / ".cache" / "proimages" / "models"))


def ensure_model(repo_id: str) -> Path:
    from huggingface_hub import snapshot_download

    local_dir = MODELS_DIR / repo_id.replace("/", "--")
    if local_dir.exists():
        return local_dir

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo_id, local_dir=local_dir)
    return local_dir
