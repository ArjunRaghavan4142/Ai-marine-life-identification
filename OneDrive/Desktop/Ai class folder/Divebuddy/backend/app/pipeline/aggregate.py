"""Turns raw per-frame detections into a biodiversity summary.

Two things happen here that the raw detection list can't give you on its own:
  - duplicate-count reduction: a track (one animal followed across frames)
    counts once per species, not once per frame it appeared in
  - low-confidence review: a track that never crossed its model's accept
    threshold in any frame goes into a review queue instead of being counted
"""
from collections import Counter, defaultdict

COMMON_NAMES: dict[str, str] = {
    "fish": "Unidentified Fish",
    "serranidae": "Grouper",
    "urchin": "Sea Urchin",
    "scaridae": "Parrotfish",
    "chaetodontidae": "Butterflyfish",
    "giant_clam": "Giant Clam",
    "lutjanidae": "Snapper",
    "muraenidae": "Moray Eel",
    "sea_cucumber": "Sea Cucumber",
    "haemulidae": "Grunt",
    "lobster": "Lobster",
    "crown_of_thorns": "Crown-of-Thorns Starfish",
    "bolbometopon_muricatum": "Bumphead Parrotfish",
    "cheilinus_undulatus": "Napoleon Wrasse",
    "cromileptes_altivelis": "Humpback Grouper",
    "ray": "Ray",
    "shark": "Shark",
    "turtle": "Sea Turtle",
}


def common_name(model_name: str) -> str:
    return COMMON_NAMES.get(model_name, model_name)


def _merge_fragmented_tracks(
    tracks: dict[str, list[dict]], gap_threshold_sec: float = 15.0
) -> dict[str, list[dict]]:
    """Merge tracks of the same species that are sequential and close in time.

    The tracker loses a fish when the camera pans away or the fish briefly
    leaves frame, then assigns a new track ID when it reappears.  If two
    tracks share the same majority species, don't overlap in time, and the
    gap between them is <= gap_threshold_sec, they're almost certainly the
    same animal — merge them under one track ID.  Tracks that overlap in
    time are genuinely different animals and are left alone.
    """
    species_groups: dict[str, list[tuple[str, list[dict]]]] = defaultdict(list)
    for track_id, dets in tracks.items():
        species = Counter(d["species"] for d in dets).most_common(1)[0][0]
        species_groups[species].append((track_id, dets))

    merged: dict[str, list[dict]] = {}
    for _species, track_list in species_groups.items():
        track_list.sort(key=lambda t: min(d["timestamp_sec"] for d in t[1]))

        cur_id, cur_dets = track_list[0]
        cur_end = max(d["timestamp_sec"] for d in cur_dets)

        for track_id, dets in track_list[1:]:
            t_start = min(d["timestamp_sec"] for d in dets)
            t_end = max(d["timestamp_sec"] for d in dets)

            if t_start > cur_end and (t_start - cur_end) <= gap_threshold_sec:
                cur_dets = cur_dets + dets
                cur_end = max(cur_end, t_end)
            else:
                merged[cur_id] = cur_dets
                cur_id, cur_dets = track_id, dets
                cur_end = t_end

        merged[cur_id] = cur_dets

    return merged


def aggregate_tracks(detections: list[dict], rarity_map: dict[str, str]) -> dict:
    raw_tracks: dict[str, list[dict]] = defaultdict(list)
    for detection in detections:
        raw_tracks[detection["track_id"]].append(detection)

    tracks = _merge_fragmented_tracks(raw_tracks)

    species_counts: Counter = Counter()
    species_conf_sum: dict[str, float] = defaultdict(float)
    species_conf_n: dict[str, int] = defaultdict(int)
    review_queue: list[dict] = []

    for track_id, track_detections in tracks.items():
        accept_conf = track_detections[0]["accept_conf"]
        accepted = [d for d in track_detections if d["confidence"] >= accept_conf]

        if accepted:
            species = Counter(d["species"] for d in accepted).most_common(1)[0][0]
            avg_conf = sum(d["confidence"] for d in accepted) / len(accepted)
            species_counts[species] += 1
            species_conf_sum[species] += avg_conf
            species_conf_n[species] += 1
        else:
            best = max(track_detections, key=lambda d: d["confidence"])
            review_queue.append({
                "track_id": track_id,
                "species_guess": best["species"],
                "max_confidence": round(best["confidence"], 3),
                "timestamp_sec": best["timestamp_sec"],
                "crop_path": best["crop_path"],
            })

    species_summary = [
        {
            "species": common_name(species),
            "unique_count": count,
            "avg_confidence": round(species_conf_sum[species] / species_conf_n[species], 3),
            "rarity": rarity_map.get(species, "unknown"),
        }
        for species, count in species_counts.most_common()
    ]

    for item in review_queue:
        item["species_guess"] = common_name(item["species_guess"])

    review_queue.sort(key=lambda r: r["timestamp_sec"])
    return {"species_summary": species_summary, "review_queue": review_queue}
