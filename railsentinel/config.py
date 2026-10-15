"""Runtime configuration, read from environment variables and an optional `.env` file."""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
STATIC_DATA_DIR = PACKAGE_DIR / "data"

# India Standard Time has no daylight saving, so a fixed offset is exact and
# avoids depending on an OS time-zone database (missing on some Windows setups).
IST = timezone(timedelta(hours=5, minutes=30), name="IST")

ROLE_LEVELS = {"passenger": 1, "operator": 2, "admin": 3}

# Demo keys keep a fresh clone working with zero configuration.
# Replace them through RS_API_KEYS before exposing the API publicly.
DEFAULT_API_KEYS = "passenger:rs-passenger-demo,operator:rs-operator-demo,admin:rs-admin-demo"
DEFAULT_PASSPORT_SECRET = "rs-demo-passport-secret-change-me"

load_dotenv(REPO_ROOT / ".env", override=False)


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_keys(raw: str) -> dict[str, str]:
    """Parse `role:key,role:key` into {key: role}."""
    keys: dict[str, str] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item or ":" not in item:
            continue
        role, key = item.split(":", 1)
        role, key = role.strip(), key.strip()
        if role in ROLE_LEVELS and key:
            keys[key] = role
    return keys


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    gemini_api_key: str | None
    llm_model: str
    llm_judge_model: str
    passport_secret: str
    api_keys: dict[str, str]
    auth_enabled: bool
    supervisor_interval_s: int
    seed: int
    llm_judge_sample_rate: float
    rate_limit_per_minute: int
    bootstrap_scale: float
    llm_for_synthetic_traffic: bool
    background_traffic: bool

    @property
    def db_path(self) -> Path:
        return self.data_dir / "railsentinel.db"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def llm_enabled(self) -> bool:
        return bool(self.gemini_api_key)


def load_settings() -> Settings:
    data_dir = Path(os.getenv("RS_DATA_DIR") or (REPO_ROOT / "data")).expanduser().resolve()
    api_key = (os.getenv("GEMINI_API_KEY") or "").strip() or None
    return Settings(
        data_dir=data_dir,
        gemini_api_key=api_key,
        llm_model=os.getenv("RS_LLM_MODEL", "gemini-3.8-flash"),
        llm_judge_model=os.getenv("RS_LLM_JUDGE_MODEL", "gemini-3.5-flash-lite"),
        passport_secret=os.getenv("RS_PASSPORT_SECRET", DEFAULT_PASSPORT_SECRET),
        api_keys=_parse_keys(os.getenv("RS_API_KEYS", DEFAULT_API_KEYS)),
        auth_enabled=_as_bool(os.getenv("RS_AUTH_ENABLED"), True),
        supervisor_interval_s=int(os.getenv("RS_SUPERVISOR_INTERVAL", "45")),
        seed=int(os.getenv("RS_SEED", "7")),
        llm_judge_sample_rate=float(os.getenv("RS_LLM_JUDGE_SAMPLE_RATE", "0.2")),
        rate_limit_per_minute=int(os.getenv("RS_RATE_LIMIT_PER_MINUTE", "240")),
        bootstrap_scale=float(os.getenv("RS_BOOTSTRAP_SCALE", "1.0")),
        llm_for_synthetic_traffic=_as_bool(os.getenv("RS_LLM_FOR_SYNTHETIC_TRAFFIC"), False),
        background_traffic=_as_bool(os.getenv("RS_BACKGROUND_TRAFFIC"), True),
    )
