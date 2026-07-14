"""Build the biodiversity summary from Gemini Vision identifications.

YOLO detection + tracking gives us tracks (WHERE animals are). This module
asks Gemini WHAT each one is, then builds the species summary at the coarse
"group" level (Parrotfish, Angelfish, Sea Turtle, ...) so many species of the
same familiar group collapse into a single reported line, and each distinct
track counts as one sighting.

Robustness comes from MULTI-CROP VOTING: a track is judged on several of its
clearest frames, not one. Each crop casts a confidence-weighted vote for a
group; the winning group wins the track. A single blurry frame that Gemini
calls "Unknown" can no longer sink a track that other frames identify clearly.

Two sources feed the summary:
  - accepted tracks (track_details): strong, multi-frame, high-confidence
    detections -- always counted (unless the winning group is excluded).
  - review-queue tracks: weak/borderline. Only promoted into the summary if
    the vote is confident AND the track persisted across several frames, so a
    single-frame false positive can never become a reported species.
"""
from app.pipeline.vision_id import identify_species

# Groups that are never reported as a "species" sighting.
EXCLUDE_GROUPS = {"Unknown", "Diver"}

# A review-queue track must persist at least this many frames before a confident
# vote can promote it into the summary.
MIN_REVIEW_FRAMES = 3

# How many of a track's top crops to put to Gemini for the vote. Kept low
# because the free Gemini tier is ~20 requests/day/model (see vision_id).
# Accepted tracks are few, so they can afford more crops for a robust vote;
# review tracks are many, so they stay cheap.
CROPS_PER_ACCEPTED = 3
CROPS_PER_REVIEW = 2

# Large solitary animals are, in a short dive clip, almost always a single
# individual -- but the tracker/multiple detectors routinely fragment one turtle
# or shark into several tracks. For these groups we report one individual per
# clip. Trade-off: genuinely-multiple megafauna in one clip are under-counted;
# acceptable because that is rare and over-counting one animal as 3 is worse.
SOLITARY_MEGAFAUNA = {"Sea Turtle", "Shark", "Ray", "Manta Ray"}

_CONF_WEIGHT = {"high": 3.0, "medium": 1.5, "low": 0.5}


def _crops_for(item, output_dir, k):
    tc = item.get("top_crops") or []
    if not tc:
        single = item.get("best_crop") or item.get("crop_path")
        tc = [single] if single else []
    return [str(output_dir / c) for c in tc[:k] if c]


def _vote(crop_ids):
    """Given per-crop id dicts for one track, pick the winning group.

    Returns (group, conf_level, common_name, win_weight, rival_weight, win_count, rival_count).
    Each crop's vote weight is its identifier's numeric score when available
    (iNaturalist's `_score`, 0-100), else a bucket weight from the confidence
    label (Gemini). Weighting by score lets a confident correct frame outvote a
    weak inconsistent one -- important because iNat often names the same fish
    differently frame-to-frame. "Unknown"/"Diver" never win unless nothing else did.
    """
    weights: dict[str, float] = {}
    counts: dict[str, int] = {}
    best_for_group: dict[str, tuple] = {}  # group -> (weight_of_best_crop, conf_level, common_name)
    for gid in crop_ids:
        group = gid.get("group", "Unknown")
        conf = gid.get("confidence", "low")
        w = gid.get("_score")
        if w is None:
            w = _CONF_WEIGHT.get(conf, 0.5)
        weights[group] = weights.get(group, 0.0) + w
        counts[group] = counts.get(group, 0) + 1
        prev = best_for_group.get(group)
        if prev is None or w > prev[0]:
            best_for_group[group] = (w, conf, gid.get("common_name", group))

    real = {g: w for g, w in weights.items() if g not in EXCLUDE_GROUPS}
    if not real:
        return "Unknown", "low", "Unknown", 0.0, 0.0, 0, 0

    group = max(real, key=real.get)
    _, conf_level, common = best_for_group[group]
    win_weight = weights[group]
    rival_weight = max((w for g, w in real.items() if g != group), default=0.0)
    win_count = counts[group]
    rival_count = max((counts[g] for g in real if g != group), default=0)
    return group, conf_level, common, win_weight, rival_weight, win_count, rival_count


def reidentify(result, output_dir, api_key, rarity_map, identify=identify_species, method="gemini-vision"):
    """Replace YOLO labels with vision-model group-level identifications in-place.

    `identify` is the per-crop identifier (Gemini or iNaturalist); `method` is the
    label recorded on the result so the report says which engine was used.
    """
    tracks = result.get("track_details", [])
    reviews = result.get("review_queue", [])

    items = []  # (kind, item, [crop_abs_paths])
    all_crops: list[str] = []
    skipped_reviews = []
    for t in tracks:
        crops = _crops_for(t, output_dir, CROPS_PER_ACCEPTED)
        items.append(("track", t, crops))
        all_crops.extend(crops)
    for r in reviews:
        # A review track that persisted fewer than MIN_REVIEW_FRAMES frames can
        # never be promoted to the summary, so don't spend Gemini quota on it --
        # leave it in the review queue with its YOLO guess.
        frames = r.get("frame_span", r.get("detection_count", 1))
        if frames < MIN_REVIEW_FRAMES:
            skipped_reviews.append(r)
            continue
        crops = _crops_for(r, output_dir, CROPS_PER_REVIEW)
        items.append(("review", r, crops))
        all_crops.extend(crops)

    if not all_crops:
        result["identification_method"] = method
        return result

    # Query each distinct crop once; map back per item.
    distinct = list(dict.fromkeys(all_crops))
    ids = identify(distinct, api_key)
    id_by_crop = dict(zip(distinct, ids))

    groups: dict[str, dict] = {}
    new_review = list(skipped_reviews)  # low-frame reviews pass through unidentified

    def add(group, conf_value):
        g = groups.setdefault(group, {"unique_count": 0, "conf_sum": 0.0, "conf_n": 0})
        g["unique_count"] += 1
        g["conf_sum"] += conf_value
        g["conf_n"] += 1

    for kind, item, crops in items:
        crop_ids = [id_by_crop[c] for c in crops if c in id_by_crop]
        group, conf_level, common, win_w, rival_w, win_count, rival_count = (
            _vote(crop_ids) if crop_ids else ("Unknown", "low", "Unknown", 0.0, 0.0, 0, 0))
        item["gemini_common"] = common
        item["gemini_group"] = group
        item["gemini_confidence"] = conf_level

        if kind == "track":
            item["species"] = common  # keep the specific name on the detail row
            if group not in EXCLUDE_GROUPS:
                add(group, item.get("best_confidence", 0.9))
        else:  # review
            item["species_guess"] = common
            frames = item.get("frame_span", item.get("detection_count", 1))
            # Promote a weak (review) track only if it is persistent, confidently
            # labelled, and its winning group OUTWEIGHS any competing real group
            # (score-weighted, so one strong correct frame beats an inconsistent
            # weak one -- key for iNaturalist, which names the same fish
            # differently frame to frame). A megafauna claim (turtle/shark/ray)
            # still needs >=2 agreeing crops, so a lone spurious frame can't
            # promote itself.
            megafauna_ok = group not in SOLITARY_MEGAFAUNA or win_count >= 2
            if (group not in EXCLUDE_GROUPS
                    and conf_level in ("high", "medium")
                    and frames >= MIN_REVIEW_FRAMES
                    and win_w > rival_w
                    and megafauna_ok):
                add(group, item.get("max_confidence", 0.6))
            else:
                new_review.append(item)

    summary = []
    for group, g in groups.items():
        avg = g["conf_sum"] / g["conf_n"] if g["conf_n"] else 0.0
        count = g["unique_count"]
        # One large solitary animal fragmented across several tracks/models is
        # still one animal -- report a single individual for these groups.
        if group in SOLITARY_MEGAFAUNA:
            count = 1
        summary.append({
            "species": group,
            "unique_count": count,
            "avg_confidence": round(avg, 3),
            "rarity": rarity_map.get(group, "unknown"),
            "is_school": False,
        })
    summary.sort(key=lambda s: s["unique_count"], reverse=True)

    result["species_summary"] = summary
    result["review_queue"] = new_review
    result["identification_method"] = method
    return result


# ---------------------------------------------------------------------------
# Offline (no-Gemini) group naming
#
# Names come straight from the YOLO models. Species-level YOLO labels are often
# wrong, but the FAMILY-level labels are reliable, so we collapse every label to
# a coarse common group by keyword. That trades species precision for a robust,
# instant, fully-offline identification.
# ---------------------------------------------------------------------------

# Ordered (keyword, group): first substring match wins, so specific tokens must
# come before generic ones. Matched case-insensitively against the common name.
_GROUP_KEYWORDS = [
    ("moorish idol", "Moorish Idol"), ("bannerfish", "Moorish Idol"),
    ("parrotfish", "Parrotfish"),
    ("angelfish", "Angelfish"),
    ("butterflyfish", "Butterflyfish"),
    ("surgeonfish", "Surgeonfish"), ("unicornfish", "Surgeonfish"),
    ("tang", "Surgeonfish"), ("bristletooth", "Surgeonfish"),
    ("triggerfish", "Triggerfish"),
    ("filefish", "Filefish"), ("leatherjacket", "Filefish"),
    ("anemonefish", "Clownfish"), ("clownfish", "Clownfish"), ("clown ", "Clownfish"),
    ("damselfish", "Damselfish"), ("chromis", "Damselfish"), ("sergeant", "Damselfish"),
    ("dascyllus", "Damselfish"), ("demoiselle", "Damselfish"),
    ("wrasse", "Wrasse"), ("hogfish", "Wrasse"), ("tuskfish", "Wrasse"), ("coris", "Wrasse"),
    ("grouper", "Grouper"), ("hind", "Grouper"), ("coral trout", "Grouper"),
    ("coral grouper", "Grouper"), ("lyretail", "Grouper"), ("soapfish", "Grouper"),
    ("snapper", "Snapper"), ("jobfish", "Snapper"), ("fusilier", "Fusilier"),
    ("sweetlips", "Sweetlips"), ("grunt", "Sweetlips"), ("rubberlip", "Sweetlips"),
    ("goatfish", "Goatfish"),
    ("emperor", "Emperor"), ("bream", "Bream"), ("monocle", "Bream"),
    ("cardinalfish", "Cardinalfish"),
    ("squirrelfish", "Squirrelfish"), ("soldierfish", "Squirrelfish"),
    ("pufferfish", "Pufferfish"), ("puffer", "Pufferfish"), ("toby", "Pufferfish"),
    ("porcupinefish", "Pufferfish"), ("boxfish", "Boxfish"),
    ("batfish", "Batfish"), ("spadefish", "Batfish"),
    ("barracuda", "Barracuda"),
    ("trevally", "Trevally"), ("jack", "Trevally"), ("pompano", "Trevally"),
    ("amberjack", "Trevally"), ("runner", "Trevally"),
    ("moray", "Moray Eel"), ("eel", "Moray Eel"),
    ("lionfish", "Lionfish"), ("scorpionfish", "Scorpionfish"), ("stonefish", "Scorpionfish"),
    ("goby", "Goby"), ("blenny", "Blenny"), ("anthias", "Anthias"), ("goldie", "Anthias"),
    ("turtle", "Sea Turtle"),
    ("shark", "Shark"),
    ("manta", "Ray"), ("stingray", "Ray"), ("ray", "Ray"),
    ("urchin", "Sea Urchin"), ("cucumber", "Sea Cucumber"),
    ("starfish", "Starfish"), ("crown-of-thorns", "Starfish"), ("sea star", "Starfish"),
    ("crab", "Crab"), ("lobster", "Lobster"), ("clam", "Giant Clam"),
    ("coral", "Coral"), ("anemone", "Anemone"),
]


def group_of(name: str) -> str:
    """Collapse a YOLO common name to a coarse common group (or keep it as-is)."""
    low = (name or "").lower()
    for kw, grp in _GROUP_KEYWORDS:
        if kw in low:
            return grp
    return name or "Unknown"


def regroup_yolo(result, rarity_map, min_review_frames=5, min_review_conf=0.80):
    """Build the group-level summary from YOLO labels only -- no Gemini.

    Accepted tracks always count. A weaker review track is promoted into the
    report only if it BOTH persisted >= `min_review_frames` frames AND hit
    >= `min_review_conf` confidence in at least one frame -- persistence alone
    lets confident-looking junk (a 0.66 "lionfish" seen for 10 frames) through,
    so the confidence floor trims that while keeping genuine sightings the models
    were fairly sure about but that never sustained the 0.85 accepted bar. Large
    solitary animals collapse to one individual.
    """
    tracks = result.get("track_details", [])
    reviews = result.get("review_queue", [])

    groups: dict[str, dict] = {}
    kept_review = []

    def add(group, conf):
        g = groups.setdefault(group, {"unique_count": 0, "conf_sum": 0.0, "conf_n": 0})
        g["unique_count"] += 1
        g["conf_sum"] += conf
        g["conf_n"] += 1

    for t in tracks:
        grp = group_of(t.get("species", ""))
        t["group"] = grp
        if grp not in EXCLUDE_GROUPS:
            add(grp, t.get("best_confidence", 0.9))

    for r in reviews:
        grp = group_of(r.get("species_guess", ""))
        r["group"] = grp
        frames = r.get("frame_span", r.get("detection_count", 1))
        max_conf = r.get("max_confidence", 0.0)
        if grp not in EXCLUDE_GROUPS and frames >= min_review_frames and max_conf >= min_review_conf:
            add(grp, max_conf)
        else:
            kept_review.append(r)

    summary = []
    for grp, g in groups.items():
        count = 1 if grp in SOLITARY_MEGAFAUNA else g["unique_count"]
        summary.append({
            "species": grp,
            "unique_count": count,
            "avg_confidence": round(g["conf_sum"] / g["conf_n"], 3) if g["conf_n"] else 0.0,
            "rarity": rarity_map.get(grp, "unknown"),
            "is_school": False,
        })
    summary.sort(key=lambda s: s["unique_count"], reverse=True)

    result["species_summary"] = summary
    result["review_queue"] = kept_review
    result["identification_method"] = "yolo-offline"
    return result
