"""
Record the software/hardware environment for reproducibility.

    python src/record_environment.py

Writes results/environment.json (Python, PyTorch, CUDA, GPU, key package
versions, git commit, SHA-256 of every checkpoint present).
"""

import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata

import torch

import htr_common as hc

PACKAGES = ["torch", "torchvision", "datasets", "numpy", "pandas",
            "matplotlib", "Pillow", "streamlit", "jiwer", "pytest"]


def version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def git(*args):
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return None


def main():
    checkpoints = {}

    for folder in ("checkpoints", "models"):
        if os.path.isdir(folder):
            for name in sorted(os.listdir(folder)):
                if name.endswith(".pt"):
                    path = os.path.join(folder, name)
                    checkpoints[path] = {
                        "sha256": hc.file_sha256(path),
                        "bytes": os.path.getsize(path),
                    }

    info = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "packages": {p: version(p) for p in PACKAGES},
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "checkpoints": checkpoints,
        "dataset": "to-be/OpenHand-Synth (Hugging Face), Indonesian 'ind' subset",
    }

    os.makedirs("results", exist_ok=True)

    with open(os.path.join("results", "environment.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
