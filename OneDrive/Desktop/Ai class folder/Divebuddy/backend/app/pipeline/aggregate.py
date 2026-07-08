"""Turns raw per-frame detections into a biodiversity summary.

Two things happen here that the raw detection list can't give you on its own:
  - duplicate-count reduction: a track (one animal followed across frames)
    counts once per species, not once per frame it appeared in
  - low-confidence review: a track that never crossed its model's accept
    threshold in any frame goes into a review queue instead of being counted
"""
from collections import Counter, defaultdict


def aggregate_tracks(detections: list[dict], rarity_map: dict[str, str]) -> dict:
    tracks: dict[str, list[dict]] = defaultdict(list)
    for detection in detections:
        tracks[detection["track_id"]].append(detection)

    species_counts: Counter = Counter()
    species_conf_sum: dict[str, float] = defaultdict(float)
    species_conf_n: dict[str, int] = defaultdict(int)
    review_queue: list[dict] = []

    for track_id, track_detections in tracks.items():
        accept_conf = track_detections[0]["accept_conf"]
        accepted = [d for d in track_detections if d["confidence"] >= accept_conf]

        if accepted:
            # A track can flicker between similar-looking classes across frames;
            # go with whichever species label won the majority of accepted frames.
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
            "species": species,
            "unique_count": count,
            "avg_confidence": round(species_conf_sum[species] / species_conf_n[species], 3),
            "rarity": rarity_map.get(species, "unknown"),
        }
        for species, count in species_counts.most_common()
    ]

    review_queue.sort(key=lambda r: r["timestamp_sec"])
    return {"species_summary": species_summary, "review_queue": review_queue}
