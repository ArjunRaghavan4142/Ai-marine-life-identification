"""Fetches the pretrained marine-detect YOLOv8 checkpoints used by DiveBuddy's
detection stage. Source: https://github.com/Orange-OpenSource/marine-detect (AGPL-3.0).
"""
import pathlib
import urllib.request

MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent / "backend" / "app" / "models"

MODEL_URLS = {
    "FishInv.pt": (
        "https://stpubtenakanclyw.blob.core.windows.net/marine-detect/models2025/FishInv.pt"
        "?sv=2022-11-02&ss=bf&srt=co&sp=rltf&se=2099-12-31T18:55:46Z"
        "&st=2025-02-03T10:55:46Z&spr=https,http&sig=w%2FTQzrECsYsjtkBXNnnuFtn%2BC06PkjgLxDgRw%2FaUUKI%3D"
    ),
    "MegaFauna.pt": (
        "https://stpubtenakanclyw.blob.core.windows.net/marine-detect/models2025/MegaFauna.pt"
        "?sv=2022-11-02&ss=bf&srt=co&sp=rltf&se=2099-12-31T18:55:46Z"
        "&st=2025-02-03T10:55:46Z&spr=https,http&sig=w%2FTQzrECsYsjtkBXNnnuFtn%2BC06PkjgLxDgRw%2FaUUKI%3D"
    ),
}


def download(name: str, url: str) -> None:
    dest = MODELS_DIR / name
    if dest.exists():
        print(f"{name} already present, skipping")
        return
    print(f"Downloading {name} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"Saved {dest} ({dest.stat().st_size / 1_000_000:.1f} MB)")


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in MODEL_URLS.items():
        download(name, url)


if __name__ == "__main__":
    main()
