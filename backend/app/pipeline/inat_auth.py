"""iNaturalist auth with automatic token refresh.

The iNaturalist API JWT (used by the vision endpoint) expires ~24h. To avoid a
daily manual regeneration, this mints a fresh JWT automatically from OAuth
credentials:

    OAuth password grant  ->  access_token  ->  GET /users/api_token  ->  JWT (24h)

The JWT is cached and re-minted only when it is near expiry (or on a 401), so
normal use makes at most a couple of auth calls per day.

CREDENTIALS -- read from environment (never hard-coded, never committed):
    INAT_CLIENT_ID, INAT_CLIENT_SECRET   (from an iNat OAuth application)
    INAT_USERNAME, INAT_PASSWORD          (your iNat login)
For convenience they may live in a gitignored file `backend/.inat_env`
(KEY=VALUE per line), which is loaded into the environment on import.

If OAuth creds are absent, callers fall back to a static INAT_TOKEN (the old
manual-24h path), so nothing breaks if auto-refresh isn't configured.
"""
import base64
import json
import os
import pathlib
import time

import requests

OAUTH_URL = "https://www.inaturalist.org/oauth/token"
API_TOKEN_URL = "https://www.inaturalist.org/users/api_token"

_ENV_FILE = pathlib.Path(__file__).resolve().parents[2] / ".inat_env"
_OAUTH_KEYS = ("INAT_CLIENT_ID", "INAT_CLIENT_SECRET", "INAT_USERNAME", "INAT_PASSWORD")

_cache = {"jwt": None, "exp": 0.0}


def _load_env_file():
    """Populate os.environ from backend/.inat_env if present (no overwrite)."""
    if not _ENV_FILE.exists():
        return
    for line in _ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env_file()


def _have_oauth_creds() -> bool:
    return all(os.environ.get(k) for k in _OAUTH_KEYS)


def _decode_exp(jwt: str) -> float:
    """Unix-exp from a JWT payload (no signature check -- just to time refreshes)."""
    try:
        payload = jwt.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return float(json.loads(base64.urlsafe_b64decode(payload)).get("exp", 0))
    except Exception:
        return 0.0


def _mint_jwt() -> str:
    r = requests.post(OAUTH_URL, data={
        "client_id": os.environ["INAT_CLIENT_ID"],
        "client_secret": os.environ["INAT_CLIENT_SECRET"],
        "grant_type": "password",
        "username": os.environ["INAT_USERNAME"],
        "password": os.environ["INAT_PASSWORD"],
    }, timeout=30)
    r.raise_for_status()
    access = r.json()["access_token"]
    r2 = requests.get(API_TOKEN_URL, headers={"Authorization": f"Bearer {access}"}, timeout=30)
    r2.raise_for_status()
    return r2.json()["api_token"]


def get_token(fallback: str = "", force: bool = False) -> str:
    """Return a currently-valid iNaturalist JWT.

    Uses OAuth auto-refresh when creds are configured; otherwise returns the
    `fallback` static token (manual path). `force=True` bypasses the cache.
    """
    now = time.time()
    if not _have_oauth_creds():
        return fallback or os.environ.get("INAT_TOKEN", "")

    if not force and _cache["jwt"] and now < _cache["exp"] - 300:
        return _cache["jwt"]

    jwt = _mint_jwt()
    exp = _decode_exp(jwt)
    _cache["jwt"] = jwt
    _cache["exp"] = exp if exp > now else now + 20 * 3600  # assume ~20h if unparseable
    print(f"  iNat auth: minted fresh token (valid ~{max(0, int((_cache['exp']-now)/3600))}h)")
    return jwt


def auto_refresh_enabled() -> bool:
    return _have_oauth_creds()
