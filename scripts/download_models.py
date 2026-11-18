"""Download the two encoders into ./models.

Default: Hugging Face Hub (``openai/clip-vit-base-patch16``, ``sentence-transformers/all-MiniLM-L6-v2``).
``--from-docker``: extract the same weights from public Weaviate inference images on Docker Hub, for
networks where huggingface.co is blocked (this is how the published results were produced).
"""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

MODELS = Path(__file__).resolve().parents[1] / "models"
HF = {
    "clip-vit-base-patch16": "openai/clip-vit-base-patch16",
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
}
DOCKER = {
    "clip-vit-base-patch16": (
        "semitechnologies/multi2vec-clip",
        "openai-clip-vit-base-patch16",
        ["app/models/openai_clip/", "app/models/openai_clip_processor/"],
    ),
    "all-MiniLM-L6-v2": (
        "semitechnologies/transformers-inference",
        "sentence-transformers-all-MiniLM-L6-v2",
        ["app/models/model/"],
    ),
}


def from_hf(name: str) -> None:
    from transformers import AutoModel, AutoProcessor, AutoTokenizer

    out = MODELS / name
    AutoModel.from_pretrained(HF[name]).save_pretrained(out)
    if name.startswith("clip"):
        AutoProcessor.from_pretrained(HF[name]).save_pretrained(out)
    else:
        AutoTokenizer.from_pretrained(HF[name]).save_pretrained(out)


def _get(url: str, headers: dict | None = None):
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=600)


def from_docker(name: str) -> None:
    repo, tag, prefixes = DOCKER[name]
    token = json.load(_get(f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull"))
    auth = {"Authorization": f"Bearer {token['token']}"}
    accept = (
        "application/vnd.oci.image.index.v1+json,application/vnd.docker.distribution.manifest.list.v2+json,"
        "application/vnd.docker.distribution.manifest.v2+json,application/vnd.oci.image.manifest.v1+json"
    )
    m = json.load(_get(f"https://registry-1.docker.io/v2/{repo}/manifests/{tag}", {**auth, "Accept": accept}))
    if "manifests" in m:
        d = next(x for x in m["manifests"] if x.get("platform", {}).get("architecture") == "amd64")["digest"]
        m = json.load(_get(f"https://registry-1.docker.io/v2/{repo}/manifests/{d}", {**auth, "Accept": accept}))
    out = MODELS / name
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for layer in sorted(m["layers"], key=lambda x: x["size"]):  # model layers are the small ones
            blob = _get(f"https://registry-1.docker.io/v2/{repo}/blobs/{layer['digest']}", auth)
            with tarfile.open(fileobj=blob, mode="r|*") as tf:
                for member in tf:
                    if member.isfile() and any(member.name.startswith(p) for p in prefixes):
                        tf.extract(member, tmp, filter="data")
            found = [p for p in prefixes if (Path(tmp) / p).exists()]
            if len(found) == len(prefixes):
                break
        for p in prefixes:
            for f in (Path(tmp) / p).iterdir():
                shutil.copy(f, out / f.name)
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-docker", action="store_true")
    args = ap.parse_args()
    for n in HF:
        (from_docker if args.from_docker else from_hf)(n)
