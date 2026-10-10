"""Train a BioCLIP prototype classifier from your own labelled crops.

This is the highest-accuracy lever and it needs very little data: BioCLIP already
knows what organisms look like, so we don't retrain it -- we just compute one
average embedding ("prototype") per species from your labelled examples. At
inference each crop is matched to the nearest prototype. A few hundred crops
across your target species is enough to sharply beat the zero-shot defaults,
because the prototypes are anchored to *your* water, lighting, and camera.

DATA LAYOUT -- one folder per species, folder name = the label you want shown:

    backend/datasets/reef_labeled/
        Clownfish/               img001.jpg  img002.jpg  ...
        Whitetip Reef Shark/     ...
        Powder Blue Tang/        ...
        Not a fish/              ...   (optional: a negative/background class)

Crops from the review queue (backend/outputs/<job>/crops/*.jpg) are ideal
training data once you've eyeballed and sorted them into the right folders.

RUN (from backend/, with the project venv):

    ../.venv/Scripts/python.exe scripts/build_bioclip_prototypes.py

It writes app/models/bioclip_prototypes.npz. The API auto-loads it on the next
restart -- no code change needed -- and prints
"loaded trained prototypes for N classes" on the first identification.
"""
import pathlib
import sys

import numpy as np
import torch
import open_clip
from PIL import Image

BACKEND = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND / "datasets" / "reef_labeled"
OUT_PATH = BACKEND / "app" / "models" / "bioclip_prototypes.npz"
MODEL_NAME = "hf-hub:imageomics/bioclip"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_PER_CLASS = 3  # a prototype from 1-2 crops is too noisy to trust

sys.path.insert(0, str(BACKEND))
from app.pipeline.identify import group_of  # noqa: E402  (map species -> coarse group)


def main():
    if not DATA_DIR.exists():
        print(f"No dataset at {DATA_DIR}\nCreate one folder per species (see this "
              f"script's docstring) and drop labelled crops inside.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading BioCLIP on {device} ...")
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME)
    model.eval().to(device)

    names, groups, protos = [], [], []
    for class_dir in sorted(p for p in DATA_DIR.iterdir() if p.is_dir()):
        imgs = [p for p in class_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
        if len(imgs) < MIN_PER_CLASS:
            print(f"  skip '{class_dir.name}': only {len(imgs)} img(s), need >= {MIN_PER_CLASS}")
            continue
        embs = []
        for p in imgs:
            try:
                x = preprocess(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
                with torch.no_grad():
                    f = model.encode_image(x)
                    f = f / f.norm(dim=-1, keepdim=True)
                embs.append(f[0].cpu())
            except Exception as e:  # noqa: BLE001
                print(f"    bad image {p.name}: {str(e)[:50]}")
        if not embs:
            continue
        proto = torch.stack(embs).mean(0)
        proto = proto / proto.norm()
        names.append(class_dir.name)
        groups.append(group_of(class_dir.name))
        protos.append(proto.numpy())
        print(f"  {class_dir.name:32s} <- {len(embs):3d} crops   (group: {groups[-1]})")

    if not names:
        print("No class had enough images -- nothing written.")
        return

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_PATH, names=np.array(names), groups=np.array(groups),
             prototypes=np.stack(protos).astype(np.float32))
    print(f"\nWrote {OUT_PATH}  ({len(names)} classes). Restart the API to use it.")


if __name__ == "__main__":
    main()
