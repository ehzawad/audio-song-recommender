from __future__ import annotations

import hashlib
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests
from tqdm.auto import tqdm

from music_similarity_rec.paths import ensure_dir


@dataclass(frozen=True, slots=True)
class DataPackage:
    name: str
    url: str
    sha1: str
    size_hint: str


FMA_PACKAGES: dict[str, DataPackage] = {
    "metadata": DataPackage(
        name="fma_metadata.zip",
        url="https://os.unil.cloud.switch.ch/fma/fma_metadata.zip",
        sha1="f0df49ffe5f2a6008d7dc83c6915b31835dfe733",
        size_hint="342 MiB",
    ),
    "small": DataPackage(
        name="fma_small.zip",
        url="https://os.unil.cloud.switch.ch/fma/fma_small.zip",
        sha1="ade154f733639d52e35e32f5593efe5be76c6d70",
        size_hint="7.2 GiB",
    ),
    "medium": DataPackage(
        name="fma_medium.zip",
        url="https://os.unil.cloud.switch.ch/fma/fma_medium.zip",
        sha1="c67b69ea232021025fca9231fc1c7c1a063ab50b",
        size_hint="22 GiB",
    ),
    "large": DataPackage(
        name="fma_large.zip",
        url="https://os.unil.cloud.switch.ch/fma/fma_large.zip",
        sha1="497109f4dd721066b5ce5e5f250ec604dc78939e",
        size_hint="93 GiB",
    ),
    "full": DataPackage(
        name="fma_full.zip",
        url="https://os.unil.cloud.switch.ch/fma/fma_full.zip",
        sha1="0f0ace23fbe9ba30ecb7e95f763e435ea802b8ab",
        size_hint="879 GiB",
    ),
}


def sha1sum(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha1()
    with Path(path).open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, dest: str | Path, overwrite: bool = False) -> Path:
    dest_path = Path(dest).expanduser().resolve()
    if dest_path.exists() and not overwrite:
        return dest_path

    ensure_dir(dest_path.parent)
    with requests.get(url, stream=True, timeout=30) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with dest_path.open("wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=f"downloading {dest_path.name}",
        ) as progress:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    progress.update(len(chunk))
    return dest_path


def unpack_zip(zip_path: str | Path, out_dir: str | Path) -> Path:
    out = ensure_dir(out_dir)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out)
    return out


def download_fma_package(
    package: str,
    out_dir: str | Path = "data/raw",
    unpack: bool = True,
    overwrite: bool = False,
) -> Path:
    if package not in FMA_PACKAGES:
        allowed = ", ".join(sorted(FMA_PACKAGES))
        raise KeyError(f"Unknown FMA package {package!r}; choose one of: {allowed}")
    pkg = FMA_PACKAGES[package]
    out_root = ensure_dir(out_dir)
    zip_path = download_file(pkg.url, out_root / pkg.name, overwrite=overwrite)
    observed = sha1sum(zip_path)
    if observed != pkg.sha1:
        raise RuntimeError(
            f"SHA-1 mismatch for {zip_path.name}: expected {pkg.sha1}, observed {observed}"
        )
    if unpack:
        unpack_zip(zip_path, out_root)
    return zip_path
