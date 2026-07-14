"""Local CLIP species identification -- offline replacement for Gemini.

Runs an open-source CLIP model (BioCLIP or OpenCLIP) locally on detection crops
and classifies each against a fixed set of marine-group labels plus background
"distractor" labels. No API, no quota, no rate limit -- the model downloads once
from HuggingFace and then runs fully offline on the GPU/CPU.

Exposes `classify_crops(crop_paths, _api_key=None)` with the SAME return shape as
the old Gemini `identify_species`, so it drops straight into `identify.reidentify`
and reuses the multi-crop voting / promotion machinery.
"""
import pathlib

import torch
import open_clip
from PIL import Image

# Loaded in order; the first that loads wins (ViT-L is strongest; BioCLIP is a
# solid cached fallback; ViT-B is the lightweight last resort).
_MODEL_CANDIDATES = [
    ("hf-hub:imageomics/bioclip", None),   # cached + species-tuned; primary
    ("ViT-L-14", "laion2b_s32b_b82k"),      # stronger, once downloaded
    ("ViT-B-32", "laion2b_s34b_b79k"),
]

_TEMPLATES = ["a photo of a {}", "underwater photo of a {}", "a {} in the ocean"]

# CLIP label -> canonical group reported in the summary.
_LABEL_TO_GROUP = {
    "moorish idol": "Moorish Idol", "bannerfish": "Moorish Idol",
    "angelfish": "Angelfish", "parrotfish": "Parrotfish",
    "butterflyfish": "Butterflyfish", "surgeonfish": "Surgeonfish",
    "tang": "Surgeonfish", "unicornfish": "Surgeonfish",
    "triggerfish": "Triggerfish", "filefish": "Filefish", "wrasse": "Wrasse",
    "damselfish": "Damselfish", "clownfish": "Clownfish", "grouper": "Grouper",
    "snapper": "Snapper", "sweetlips": "Sweetlips", "goatfish": "Goatfish",
    "fusilier": "Fusilier", "emperor fish": "Emperor", "cardinalfish": "Cardinalfish",
    "squirrelfish": "Squirrelfish", "pufferfish": "Pufferfish", "boxfish": "Boxfish",
    "batfish": "Batfish", "barracuda": "Barracuda", "trevally": "Trevally",
    "moray eel": "Moray Eel", "lionfish": "Lionfish", "scorpionfish": "Scorpionfish",
    "goby": "Goby", "blenny": "Blenny", "anthias": "Anthias",
    "sea turtle": "Sea Turtle", "shark": "Shark", "stingray": "Ray", "manta ray": "Ray",
    "sea urchin": "Sea Urchin", "sea cucumber": "Sea Cucumber", "starfish": "Starfish",
    "crab": "Crab", "lobster": "Lobster", "coral": "Coral", "sea anemone": "Anemone",
    "fish": "Fish",
}
# Background / non-organism concepts -> not a species sighting.
_DISTRACTORS = {
    "coral reef background": "Unknown", "open ocean water": "Unknown",
    "rocks and sand on the seabed": "Unknown", "a blurry underwater scene": "Unknown",
    "a scuba diver": "Diver",
}

# Softmax-probability thresholds for the winning label -> confidence bucket.
_HIGH, _MED = 0.35, 0.18

_STATE = {}


def _load():
    if _STATE:
        return _STATE
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    last_err = None
    for name, pretrained in _MODEL_CANDIDATES:
        try:
            if pretrained:
                model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrained)
                tok = open_clip.get_tokenizer(name)
            else:
                model, _, preprocess = open_clip.create_model_and_transforms(name)
                tok = open_clip.get_tokenizer(name)
            model = model.to(dev).eval()
            print(f"  CLIP identifier: loaded {name} ({pretrained or 'default'}) on {dev}")
            _build(model, preprocess, tok, dev, name)
            return _STATE
        except Exception as e:
            last_err = e
            print(f"  CLIP identifier: {name} unavailable ({str(e)[:60]})")
    raise RuntimeError(f"No CLIP model could be loaded: {last_err}")


def _build(model, preprocess, tok, dev, name):
    labels = list(_LABEL_TO_GROUP) + list(_DISTRACTORS)
    with torch.no_grad():
        embs = []
        for lab in labels:
            toks = tok([t.format(lab) for t in _TEMPLATES]).to(dev)
            e = model.encode_text(toks)
            e = e / e.norm(dim=-1, keepdim=True)
            embs.append(e.mean(0))
        txt = torch.stack(embs)
        txt = txt / txt.norm(dim=-1, keepdim=True)
    _STATE.update(model=model, preprocess=preprocess, dev=dev, labels=labels, txt=txt, name=name)


def _group_and_conf(label, prob):
    if label in _DISTRACTORS:
        return _DISTRACTORS[label], "Unknown", "low"
    group = _LABEL_TO_GROUP.get(label, "Unknown")
    conf = "high" if prob >= _HIGH else "medium" if prob >= _MED else "low"
    return group, label, conf


def classify_crops(crop_paths, _api_key=None):
    """Classify each crop locally. Returns dicts shaped like the Gemini output."""
    st = _load()
    model, preprocess, dev, labels, txt = (
        st["model"], st["preprocess"], st["dev"], st["labels"], st["txt"])

    out = []
    for i, path in enumerate(crop_paths):
        if not path or not pathlib.Path(path).exists():
            out.append({"common_name": "Unknown", "scientific_name": "", "group": "Unknown", "confidence": "low"})
            continue
        try:
            img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(dev)
            with torch.no_grad():
                f = model.encode_image(img)
                f = f / f.norm(dim=-1, keepdim=True)
                probs = (100.0 * f @ txt.T).softmax(dim=-1)[0]
            idx = int(probs.argmax())
            label, prob = labels[idx], float(probs[idx])
            group, common, conf = _group_and_conf(label, prob)
            out.append({
                "common_name": common.title() if common != "Unknown" else "Unknown",
                "scientific_name": "",
                "group": group,
                "confidence": conf,
            })
            print(f"  CLIP [{i+1}/{len(crop_paths)}]: {group} ({label} {prob:.2f} -> {conf})")
        except Exception as e:
            print(f"  CLIP failed for crop {i}: {str(e)[:80]}")
            out.append({"common_name": "Unknown", "scientific_name": "", "group": "Unknown", "confidence": "low"})
    return out
