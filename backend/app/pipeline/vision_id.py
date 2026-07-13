"""Gemini Vision species identification.

Uses Google Gemini to identify marine species from detection crops.
YOLO models handle detection (bounding boxes), Gemini handles identification.

Each crop is identified at two granularities:
  - common_name / scientific_name: species-level, as specific as the image allows
  - group: a coarse common-group label (e.g. "Parrotfish", "Angelfish",
    "Sea Turtle") used to build the biodiversity summary so that many species
    of the same familiar group collapse into one reported line.

FREE-TIER QUOTA: Gemini's free tier caps each model at ~20 requests/DAY. To get
usable headroom we ROTATE across several free models -- when one model's daily
quota is exhausted we switch to the next. A per-day 429 is NOT retried (it won't
clear for hours); only transient faults (per-minute 429, 503, network) are.
"""
import json
import pathlib
import time

from google import genai

# Tried in order; on daily-quota exhaustion we advance to the next. Fuller flash
# models first (better fine-grained ID), lite models as fallback. Keep the
# two 20/day models that may have reset at the end.
MODEL_CANDIDATES = [
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-3.5-flash",
    "gemini-flash-latest",
]

# Coarse groups the summary is built from. Gemini is asked to pick the closest
# one; this keeps "Stoplight Parrotfish" and "Bullethead Parrotfish" from
# showing up as two separate summary lines.
GROUPS = [
    "Parrotfish", "Angelfish", "Butterflyfish", "Surgeonfish", "Tang",
    "Triggerfish", "Filefish", "Wrasse", "Damselfish", "Clownfish",
    "Grouper", "Snapper", "Sweetlips", "Goatfish", "Fusilier", "Emperor",
    "Bream", "Cardinalfish", "Squirrelfish", "Pufferfish", "Boxfish",
    "Moorish Idol", "Batfish", "Barracuda", "Trevally", "Moray Eel",
    "Lionfish", "Scorpionfish", "Goby", "Blenny", "Anthias", "Fish",
    "Sea Turtle", "Shark", "Ray", "Eel",
    "Sea Urchin", "Sea Cucumber", "Starfish", "Crab", "Lobster", "Nudibranch",
    "Coral", "Anemone", "Diver", "Unknown",
]

PROMPT = """You are a marine biologist identifying species from underwater dive footage.

Look at this cropped image from a dive video and identify the marine organism.

Reply with ONLY a raw JSON object (no markdown, no code fences), in this exact format:
{"common_name": "...", "scientific_name": "...", "group": "...", "confidence": "high|medium|low"}

Rules:
- "common_name": the most specific species/common name you can justify from the image.
- "scientific_name": binomial if you can, else "".
- "group": the SINGLE closest match from this list: %s
  Pick the coarse group a layperson would use. If it is clearly not a marine
  organism (e.g. rock, sand, blur, bubbles, a hand), use "Unknown".
- "confidence": "high" only if you are quite sure; "low" if the crop is blurry,
  tiny, occluded, or ambiguous.
- If you cannot tell what it is, use group "Unknown" and confidence "low".""" % ", ".join(GROUPS)


def _parse(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0].strip()
    parsed = json.loads(text)
    return {
        "common_name": parsed.get("common_name", "Unknown") or "Unknown",
        "scientific_name": parsed.get("scientific_name", "") or "",
        "group": parsed.get("group", "Unknown") or "Unknown",
        "confidence": parsed.get("confidence", "medium") or "medium",
    }


def _classify_error(err: str) -> str:
    """daily | transient | fatal"""
    if "PerDay" in err or "limit: 0" in err:
        return "daily"
    if any(s in err for s in ("429", "503", "getaddrinfo", "Connection", "connection",
                              "Timeout", "timeout", "Temporary failure", "ServiceUnavailable",
                              "Server disconnected")):
        return "transient"
    return "fatal"


def _identify_one(client, img_bytes, state, label):
    """Identify one crop, rotating models on daily-quota exhaustion."""
    part = genai.types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")
    while state["mi"] < len(MODEL_CANDIDATES):
        model = MODEL_CANDIDATES[state["mi"]]
        for attempt in range(3):
            try:
                resp = client.models.generate_content(model=model, contents=[PROMPT, part])
                parsed = _parse(resp.text)
                print(f"  Gemini [{label}] ({model}): {parsed['common_name']} "
                      f"({parsed['group']}, {parsed['confidence']})")
                return parsed
            except Exception as e:
                kind = _classify_error(str(e))
                if kind == "daily":
                    print(f"  {model}: daily quota exhausted -> next model")
                    break  # advance model
                if kind == "transient" and attempt < 2:
                    time.sleep(8 * (attempt + 1))
                    continue
                # transient out of attempts, or fatal -> try next model
                print(f"  {model}: {str(e)[:70]} -> next model")
                break
        state["mi"] += 1
    print(f"  [{label}] all models exhausted")
    return {"common_name": "Unknown", "scientific_name": "", "group": "Unknown",
            "confidence": "low", "_error": True}


def identify_species(crop_paths: list[str], api_key: str) -> list[dict]:
    """Send each crop to Gemini Vision (with model rotation) and return ids."""
    client = genai.Client(api_key=api_key)
    state = {"mi": 0}  # index into MODEL_CANDIDATES; only advances on exhaustion

    results = []
    n = len(crop_paths)
    for i, crop_path in enumerate(crop_paths):
        if not crop_path or not pathlib.Path(crop_path).exists():
            results.append({"common_name": "Unknown", "scientific_name": "", "group": "Unknown", "confidence": "low"})
            continue
        img_bytes = pathlib.Path(crop_path).read_bytes()
        results.append(_identify_one(client, img_bytes, state, f"{i+1}/{n}"))
        # Space out requests to respect per-minute limits, but stop early if we
        # have run out of models.
        if state["mi"] < len(MODEL_CANDIDATES) and i < n - 1:
            time.sleep(4)

    return results
