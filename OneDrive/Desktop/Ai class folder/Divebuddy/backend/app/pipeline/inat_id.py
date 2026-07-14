"""iNaturalist Vision species identification.

Sends detection crops to iNaturalist's computer-vision scoring endpoint, which
is trained on the world's largest organism-photo dataset (excellent marine
coverage). Returns the same shape as the Gemini identifier, so it plugs straight
into `identify.reidentify` and reuses the multi-crop voting machinery.

AUTH: uses `inat_auth.get_token()`, which auto-refreshes a JWT from OAuth
credentials when configured (so it never expires on you), or falls back to the
static token passed as `api_key`. On a 401 mid-run it force-refreshes once.

NOTE: this CV endpoint is intended for logged-in personal use on iNaturalist's
own site, not for third-party apps at scale -- fine for testing / a class
project, but respect their rate limits and ToS.
"""
import pathlib
import time

import requests

from app.pipeline.identify import group_of
from app.pipeline import inat_auth

SCORE_URL = "https://api.inaturalist.org/v1/computervision/score_image"

# iNaturalist asks callers to stay gentle; ~1s spacing is well within limits and
# far faster than Gemini's free-tier 4s.
_SPACING = 1.2

# iNat's model spans the whole tree of life, so on a blurry crop it can pick a
# mammal/bird. Restrict to the iconic taxa that actually occur on a reef dive:
# ray-finned fishes, reptiles (sea turtles), molluscs, and the Animalia catch-all
# (sharks/rays, corals, echinoderms, crustaceans -- none of which have their own
# iconic taxon). This drops Mammalia (sea lion/dolphin/seal) and Aves (booby).
_ALLOWED_ICONIC = {"Actinopterygii", "Reptilia", "Mollusca", "Animalia"}

# combined_score (0-100) -> confidence bucket. Only medium+ crops can promote a
# review track, so _MED is the real gate. On these dive-video crops iNat cleanly
# separates confident correct IDs (turtle ~27, Moorish idols ~10-16) from
# uncertain guesses (butterflyfish/snapper/drum ~7-8), so _MED=10 keeps the
# former and drops the latter.
_HIGH, _MED = 20.0, 10.0


def _one(session, crop_path, token):
    with open(crop_path, "rb") as f:
        resp = session.post(
            SCORE_URL,
            headers={"Authorization": f"Bearer {token}"},
            files={"image": (pathlib.Path(crop_path).name, f, "image/jpeg")},
            timeout=30,
        )
    resp.raise_for_status()  # 401 surfaces here and triggers a token refresh in the caller
    results = resp.json().get("results") or []
    # Take the top-scoring result whose taxon is a reef-relevant clade.
    chosen = next((r for r in results
                   if (r.get("taxon") or {}).get("iconic_taxon_name") in _ALLOWED_ICONIC), None)
    if chosen is None:
        return {"common_name": "Unknown", "scientific_name": "", "group": "Unknown", "confidence": "low", "_score": 0.0}
    taxon = chosen.get("taxon", {}) or {}
    score = chosen.get("combined_score", chosen.get("vision_score", 0.0)) or 0.0
    common = (taxon.get("preferred_common_name") or taxon.get("name") or "Unknown")
    sci = taxon.get("name", "") if taxon.get("rank") in ("species", "genus") else ""
    conf = "high" if score >= _HIGH else "medium" if score >= _MED else "low"
    return {
        "common_name": common,
        "scientific_name": sci,
        "group": group_of(common),
        "confidence": conf,
        "_score": round(score, 1),
    }


def identify_inat(crop_paths, api_key):
    """Identify each crop via iNaturalist, auto-refreshing the token as needed.

    `api_key` is used as the static-token fallback when OAuth auto-refresh isn't
    configured (see inat_auth).
    """
    token = inat_auth.get_token(fallback=api_key)
    if not token:
        raise RuntimeError("No iNaturalist auth -- set INAT_TOKEN, or OAuth creds "
                           "(INAT_CLIENT_ID/SECRET/USERNAME/PASSWORD) for auto-refresh")
    session = requests.Session()
    out = []
    n = len(crop_paths)
    for i, path in enumerate(crop_paths):
        if not path or not pathlib.Path(path).exists():
            out.append({"common_name": "Unknown", "scientific_name": "", "group": "Unknown", "confidence": "low"})
            continue
        got = None
        refreshed = False
        for attempt in range(3):
            try:
                got = _one(session, path, token)
                break
            except requests.HTTPError as e:
                code = e.response.status_code if e.response is not None else 0
                if code == 401 and not refreshed:
                    token = inat_auth.get_token(fallback=api_key, force=True)
                    refreshed = True
                    continue
                if code == 429 and attempt < 2:
                    time.sleep(10 * (attempt + 1))
                    continue
                print(f"  iNat HTTP {code} for crop {i}: {str(e)[:60]}")
                break
            except Exception as e:
                if attempt < 2 and any(s in str(e) for s in ("Connection", "timeout", "getaddrinfo")):
                    time.sleep(5)
                    continue
                print(f"  iNat failed for crop {i}: {str(e)[:80]}")
                break
        if got is None:
            got = {"common_name": "Unknown", "scientific_name": "", "group": "Unknown", "confidence": "low", "_error": True}
        out.append(got)
        print(f"  iNat [{i+1}/{n}]: {got['common_name']} ({got['group']}, {got['confidence']}, score={got.get('_score','?')})")
        if i < n - 1:
            time.sleep(_SPACING)
    return out
