"""Local species identification with BioCLIP.

BioCLIP (imageomics/bioclip) is a CLIP model trained on the tree of life -- it
knows organisms, including strong reef-fish coverage. Running it locally makes
species naming independent of any API token: no iNaturalist 401s, no daily
quota, no 24h expiry. It is the PRIMARY identifier; iNaturalist stays on as an
optional second opinion, and the YOLO offline grouping as the final fallback.

Three accuracy levers live here:

  * ZERO-SHOT against a curated reef label set -- embed each label's (scientific)
    name once, score every crop against them, sum probability per coarse GROUP so
    several species of one family reinforce each other, and name the crop with the
    best species in the winning group. Background labels (sand/water/blur) absorb
    non-organism crops so junk routes to "Unknown".

  * REGION RESTRICTION -- each label is tagged with where it occurs. On a dive in
    a given region we drop species that don't live there, so the model can't pick
    an impossible ID and the remaining probability mass is redistributed over
    plausible species.

  * A TRAINED PROTOTYPE CLASSIFIER (optional) -- if `app/models/bioclip_prototypes.npz`
    exists (built by scripts/build_bioclip_prototypes.py from your own labelled
    crops), each crop is matched to the nearest class prototype first; zero-shot
    only fills in where the trained classifier isn't confident. This is the
    highest-accuracy path and needs only a few hundred labelled crops.

Returns the SAME dict shape as the Gemini/iNaturalist identifiers
({common_name, scientific_name, group, confidence, _score}).
"""
import os
import pathlib

# (common_name, scientific_name_or_"", group, region_tag)
#   region_tag "IP"  = Indo-Pacific species (also valid for Red Sea / Seychelles)
#   region_tag "ALL" = cosmopolitan / circumtropical family or megafauna
# Non-Indo-Pacific regions (Caribbean/Med/temperate) keep only "ALL" labels, so
# ID there honestly degrades to family level rather than asserting IP species.
REEF_LABELS = [
    # --- megafauna ---
    ("Green Sea Turtle", "Chelonia mydas", "Sea Turtle", "ALL"),
    ("Hawksbill Turtle", "Eretmochelys imbricata", "Sea Turtle", "ALL"),
    ("Blacktip Reef Shark", "Carcharhinus melanopterus", "Shark", "IP"),
    ("Whitetip Reef Shark", "Triaenodon obesus", "Shark", "IP"),
    ("Grey Reef Shark", "Carcharhinus amblyrhynchos", "Shark", "IP"),
    ("Whale Shark", "Rhincodon typus", "Shark", "ALL"),
    ("Manta Ray", "Mobula birostris", "Ray", "ALL"),
    ("Bluespotted Stingray", "Taeniura lymma", "Ray", "IP"),
    ("Eagle Ray", "Aetobatus narinari", "Ray", "ALL"),
    # --- parrotfish ---
    ("Parrotfish", "Scarus", "Parrotfish", "ALL"),
    ("Bullethead Parrotfish", "Chlorurus sordidus", "Parrotfish", "IP"),
    ("Bumphead Parrotfish", "Bolbometopon muricatum", "Parrotfish", "IP"),
    # --- wrasse ---
    ("Wrasse", "Labridae", "Wrasse", "ALL"),
    ("Napoleon Wrasse", "Cheilinus undulatus", "Wrasse", "IP"),
    ("Birdmouth Wrasse", "Gomphosus caeruleus", "Wrasse", "IP"),
    ("Cleaner Wrasse", "Labroides dimidiatus", "Wrasse", "IP"),
    # --- surgeonfish / tang / unicornfish ---
    ("Surgeonfish", "Acanthurus", "Surgeonfish", "ALL"),
    ("Powder Blue Tang", "Acanthurus leucosternon", "Surgeonfish", "IP"),
    ("Sailfin Tang", "Zebrasoma", "Surgeonfish", "IP"),
    ("Unicornfish", "Naso", "Surgeonfish", "IP"),
    ("Moorish Idol", "Zanclus cornutus", "Moorish Idol", "IP"),
    # --- butterflyfish / angelfish ---
    ("Butterflyfish", "Chaetodon", "Butterflyfish", "ALL"),
    ("Angelfish", "Pomacanthus", "Angelfish", "ALL"),
    ("Emperor Angelfish", "Pomacanthus imperator", "Angelfish", "IP"),
    ("Dwarf Angelfish", "Centropyge", "Angelfish", "ALL"),
    # --- damselfish / clownfish / anthias ---
    ("Clownfish", "Amphiprion ocellaris", "Clownfish", "IP"),
    ("Damselfish", "Pomacentridae", "Damselfish", "ALL"),
    ("Sergeant Major", "Abudefduf", "Damselfish", "ALL"),
    ("Chromis", "Chromis", "Damselfish", "ALL"),
    ("Anthias", "Pseudanthias", "Anthias", "IP"),
    # --- triggerfish / puffer / box ---
    ("Triggerfish", "Balistidae", "Triggerfish", "ALL"),
    ("Titan Triggerfish", "Balistoides viridescens", "Triggerfish", "IP"),
    ("Pufferfish", "Tetraodontidae", "Pufferfish", "ALL"),
    ("Porcupinefish", "Diodon", "Pufferfish", "ALL"),
    ("Boxfish", "Ostracion", "Boxfish", "ALL"),
    # --- grouper / snapper / sweetlips / emperor / fusilier / goatfish ---
    ("Grouper", "Epinephelus", "Grouper", "ALL"),
    ("Coral Grouper", "Cephalopholis miniata", "Grouper", "IP"),
    ("Snapper", "Lutjanus", "Snapper", "ALL"),
    ("Fusilier", "Caesio", "Fusilier", "IP"),
    ("Sweetlips", "Plectorhinchus", "Sweetlips", "IP"),
    ("Emperor", "Lethrinus", "Emperor", "IP"),
    ("Goatfish", "Mulloidichthys", "Goatfish", "ALL"),
    ("Bream", "Nemipterus", "Bream", "IP"),
    # --- specialists / predators ---
    ("Lionfish", "Pterois volitans", "Lionfish", "ALL"),  # native IP, invasive Atlantic
    ("Scorpionfish", "Scorpaenopsis", "Scorpionfish", "ALL"),
    ("Stonefish", "Synanceia verrucosa", "Scorpionfish", "IP"),
    ("Moray Eel", "Gymnothorax", "Moray Eel", "ALL"),
    ("Barracuda", "Sphyraena", "Barracuda", "ALL"),
    ("Trevally", "Caranx", "Trevally", "ALL"),
    ("Batfish", "Platax", "Batfish", "IP"),
    ("Squirrelfish", "Sargocentron", "Squirrelfish", "ALL"),
    ("Cardinalfish", "Apogon", "Cardinalfish", "ALL"),
    ("Goby", "Gobiidae", "Goby", "ALL"),
    ("Blenny", "Blenniidae", "Blenny", "ALL"),
    # --- invertebrates ---
    ("Nudibranch", "Nudibranchia", "Nudibranch", "ALL"),
    ("Giant Clam", "Tridacna", "Giant Clam", "IP"),
    ("Sea Urchin", "Diadema", "Sea Urchin", "ALL"),
    ("Sea Cucumber", "Holothuroidea", "Sea Cucumber", "ALL"),
    ("Crown-of-Thorns Starfish", "Acanthaster planci", "Starfish", "IP"),
    ("Starfish", "Asteroidea", "Starfish", "ALL"),
    ("Octopus", "Octopus", "Octopus", "ALL"),
    ("Cuttlefish", "Sepia", "Cuttlefish", "ALL"),
    ("Lobster", "Panulirus", "Lobster", "ALL"),
    ("Crab", "Brachyura", "Crab", "ALL"),
    ("Christmas Tree Worm", "Spirobranchus giganteus", "Coral", "ALL"),
    # --- sessile / reef structure ---
    ("Hard Coral", "Acropora", "Coral", "ALL"),
    ("Boulder Coral", "Porites", "Coral", "ALL"),
    ("Soft Coral", "Alcyonacea", "Coral", "ALL"),
    ("Sea Anemone", "Actiniaria", "Anemone", "ALL"),
    ("Reef Fish", "", "Fish", "ALL"),
    # --- background / negatives -> Unknown ---
    ("Sand", "", "Unknown", "ALL"),
    ("Open Water", "", "Unknown", "ALL"),
    ("Rock", "", "Unknown", "ALL"),
    ("Blurry Water", "", "Unknown", "ALL"),
    ("Diver", "", "Diver", "ALL"),
]

# Region string (from the upload form / app selector) -> region tags kept.
# Anything unmapped (global/other/None) keeps everything.
_REGION_TAGS = {
    "indo-pacific": {"IP", "ALL"}, "indopacific": {"IP", "ALL"}, "indo_pacific": {"IP", "ALL"},
    "seychelles": {"IP", "ALL"}, "gbr": {"IP", "ALL"}, "red-sea": {"IP", "ALL"}, "redsea": {"IP", "ALL"},
    "caribbean": {"ALL"}, "atlantic": {"ALL"},
    "mediterranean": {"ALL"}, "temperate": {"ALL"}, "coldwater": {"ALL"},
}

MODEL_NAME = "hf-hub:imageomics/bioclip"
MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent / "models"
PROTO_PATH = MODELS_DIR / "bioclip_prototypes.npz"

# CPU by default: the identifier runs on only a handful of track crops per video,
# so CPU is fast enough and avoids contending for VRAM with the YOLO detectors on
# small GPUs. Set BIOCLIP_DEVICE=cuda to force GPU.
_DEVICE = os.environ.get("BIOCLIP_DEVICE", "cpu")

_TEMPLATES = ("a photo of {}.", "an underwater photo of {}.")

# Summed group-probability (over the active label set) -> confidence bucket.
_HIGH, _MED = 0.45, 0.20
# A trained prototype match at/above this cosine-softmax prob overrides zero-shot.
_PROTO_MIN = 0.35

_state = {"loaded": False, "ok": False}


def available() -> bool:
    if _state["loaded"]:
        return _state["ok"]
    import importlib.util
    return importlib.util.find_spec("open_clip") is not None


def _load():
    """One-time load of model, full label text embeddings, and (if present) the
    trained prototype classifier."""
    if _state["loaded"]:
        return _state["ok"]
    _state["loaded"] = True
    try:
        import torch
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME)
        tokenizer = open_clip.get_tokenizer(MODEL_NAME)
        model.eval().to(_DEVICE)
        feats = []
        with torch.no_grad():
            for common, sci, _g, _r in REEF_LABELS:
                name = sci or common
                toks = tokenizer([t.format(name) for t in _TEMPLATES]).to(_DEVICE)
                tf = model.encode_text(toks)
                tf = tf / tf.norm(dim=-1, keepdim=True)
                feats.append(tf.mean(dim=0))
        text_feats = torch.stack(feats)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        _state.update(model=model, preprocess=preprocess, text_feats=text_feats,
                      regions=[r for *_x, r in REEF_LABELS], ok=True)
        print(f"  BioCLIP: loaded on {_DEVICE} ({len(REEF_LABELS)} reef labels)")
        _load_prototypes(torch)
    except Exception as exc:  # noqa: BLE001
        print(f"  BioCLIP unavailable ({str(exc)[:90]}) — will fall back")
        _state["ok"] = False
    return _state["ok"]


def _load_prototypes(torch):
    """Load a trained prototype classifier if one has been built."""
    _state["proto"] = None
    if not PROTO_PATH.exists():
        return
    try:
        import numpy as np
        data = np.load(PROTO_PATH, allow_pickle=True)
        protos = torch.tensor(data["prototypes"], dtype=torch.float32).to(_DEVICE)
        protos = protos / protos.norm(dim=-1, keepdim=True)
        _state["proto"] = {
            "names": list(data["names"]), "groups": list(data["groups"]), "vecs": protos,
        }
        print(f"  BioCLIP: loaded trained prototypes for {len(data['names'])} classes")
    except Exception as exc:  # noqa: BLE001
        print(f"  BioCLIP: could not load prototypes ({str(exc)[:60]})")


_UNKNOWN = {"common_name": "Unknown", "scientific_name": "", "group": "Unknown", "confidence": "low", "_score": 0.0}


def _active_idx(region):
    """Indices of REEF_LABELS valid for a dive region ("" / None => all)."""
    tags = _REGION_TAGS.get((region or "").lower().strip())
    if not tags:
        return list(range(len(REEF_LABELS)))
    return [i for i, r in enumerate(_state["regions"]) if r in tags]


def identify_bioclip(crop_paths, api_key=None, region=None):
    """Identify each crop locally with BioCLIP, restricted to `region`'s species.
    `api_key` is ignored (uniform identifier signature)."""
    if not _load():
        raise RuntimeError("BioCLIP could not be loaded")
    import torch
    from PIL import Image
    model, preprocess, text_feats = _state["model"], _state["preprocess"], _state["text_feats"]
    proto = _state.get("proto")

    idx = _active_idx(region)
    sub_feats = text_feats[idx]

    out = []
    n = len(crop_paths)
    for i, path in enumerate(crop_paths):
        if not path or not pathlib.Path(path).exists():
            out.append(dict(_UNKNOWN)); continue
        try:
            img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(_DEVICE)
            with torch.no_grad():
                feat = model.encode_image(img)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                # 1) trained prototype classifier first, if available + confident
                if proto is not None:
                    pp = (100.0 * feat @ proto["vecs"].T).softmax(dim=-1)[0]
                    j = int(pp.argmax()); pscore = float(pp[j])
                    if pscore >= _PROTO_MIN:
                        name = proto["names"][j]; grp = proto["groups"][j]
                        conf = "high" if pscore >= 0.6 else "medium"
                        out.append({"common_name": name, "scientific_name": "", "group": grp,
                                    "confidence": conf, "_score": round(pscore * 100, 1)})
                        print(f"  BioCLIP* [{i+1}/{n}]: {name} ({grp}, {conf}, score={round(pscore*100,1)})")
                        continue
                # 2) zero-shot over the region-restricted label set
                probs = (100.0 * feat @ sub_feats.T).softmax(dim=-1)[0].tolist()
        except Exception as e:  # noqa: BLE001
            print(f"  BioCLIP failed for crop {i}: {str(e)[:70]}")
            out.append(dict(_UNKNOWN)); continue

        group_prob: dict[str, float] = {}
        best_in_group: dict[str, tuple] = {}
        for k, p in zip(idx, probs):
            common, sci, group, _r = REEF_LABELS[k]
            group_prob[group] = group_prob.get(group, 0.0) + p
            if group not in best_in_group or p > best_in_group[group][0]:
                best_in_group[group] = (p, common, sci)

        win_group = max(group_prob, key=group_prob.get)
        score = group_prob[win_group]
        _, common, sci = best_in_group[win_group]
        if win_group in ("Unknown", "Diver"):
            got = {"common_name": win_group, "scientific_name": "", "group": win_group,
                   "confidence": "low", "_score": round(score * 100, 1)}
        else:
            conf = "high" if score >= _HIGH else "medium" if score >= _MED else "low"
            got = {"common_name": common, "scientific_name": sci, "group": win_group,
                   "confidence": conf, "_score": round(score * 100, 1)}
        out.append(got)
        print(f"  BioCLIP [{i+1}/{n}]: {got['common_name']} "
              f"({got['group']}, {got['confidence']}, score={got['_score']})")
    return out
