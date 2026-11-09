"""Settings from the environment (prefix CHAVRUTA_). Secrets have no defaults: the app refuses to
start without a real session secret and real passwords, instead of running with a known one."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from chavruta.llm import Mode
from chavruta.scope import Strictness

BACKEND = Path(__file__).resolve().parents[2]
_PLACEHOLDERS = {"change-me", "changeme", "secret", "password", "replace-me"}


def read_secret(value: str | None, file: str | None) -> str | None:
    """A secret from VAR or VAR_FILE (Docker secrets, or the generated-secrets volume)."""
    if file:
        return Path(file).read_text(encoding="utf-8").strip()
    return value


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CHAVRUTA_", env_file=None, extra="ignore")

    database_url: str = "postgresql+psycopg://chavruta@localhost:5432/chavruta"
    #: Password for database_url, from a file (the generated-secrets volume); inserted into the URL.
    db_password_file: str | None = None
    data_dir: Path = BACKEND / "data"
    recordings_dir: Path = BACKEND / "eval" / "recordings" / "llm"
    llm_mode: Mode = Mode.REPLAY
    strictness: Strictness = Strictness.BALANCED
    relevance_threshold: float = 0.05
    reranker: bool = True
    session_hours: int = 12
    secure_cookies: bool = False
    api_docs: bool = False
    # Shabbat closing: Hebcal location; "auto" follows Hebcal, "closed" forces the closed screen
    # (for demos and tests). There is deliberately no "always open" setting.
    geonameid: int = 281184
    israel: bool = True
    timezone: str = "Asia/Jerusalem"
    closing: str = "auto"

    session_secret: str | None = None
    session_secret_file: str | None = None
    learner_password: str | None = None
    learner_password_file: str | None = None
    admin_password: str | None = None
    admin_password_file: str | None = None

    @field_validator("closing")
    @classmethod
    def _closing(cls, v: str) -> str:
        if v not in ("auto", "closed"):
            raise ValueError("closing must be auto or closed")
        return v

    def dsn(self) -> str:
        if not self.db_password_file:
            return self.database_url
        from urllib.parse import quote

        password = quote(Path(self.db_password_file).read_text(encoding="utf-8").strip(), safe="")
        scheme, rest = self.database_url.split("://", 1)
        user, host = rest.split("@", 1)
        return f"{scheme}://{user.split(':')[0]}:{password}@{host}"

    @property
    def db_host(self) -> str:
        return urlparse(self.database_url.replace("+psycopg", "")).hostname or "localhost"

    def secret(self) -> str:
        value = read_secret(self.session_secret, self.session_secret_file)
        if not value or len(value.encode()) < 32 or value.lower() in _PLACEHOLDERS:
            raise RuntimeError(
                "CHAVRUTA_SESSION_SECRET (or _FILE) must be set to at least 32 random bytes; "
                "docker compose generates one on first run"
            )
        return value

    def password(self, role: str) -> str:
        value = read_secret(getattr(self, f"{role}_password"), getattr(self, f"{role}_password_file"))
        if not value or len(value) < 10 or value.lower() in _PLACEHOLDERS:
            raise RuntimeError(f"CHAVRUTA_{role.upper()}_PASSWORD (or _FILE) must be set (10+ characters)")
        return value
