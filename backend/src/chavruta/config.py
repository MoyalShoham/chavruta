"""Settings from the environment (prefix CHAVRUTA_). Passwords have no defaults: the app refuses to
start without real ones, instead of running with a known one. Sessions are random server-side
tokens, so there is no signing secret to configure."""

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
    #: Candidates the reranker scores per question: 20, chosen on dev (recall@10 71% against 74% at
    #: 50, for 40% of the compute). The reranker cannot be turned off, because the relevance
    #: threshold is defined on its scores.
    rerank_top: int = 20
    #: Model calls (ask, card suggestions) per account: a spend limit.
    llm_calls_per_hour: int = 60
    llm_calls_per_day: int = 300
    #: Read the client address from X-Forwarded-For. Only behind the bundled nginx, which sets it.
    trust_proxy: bool = False
    session_hours: int = 12
    secure_cookies: bool = False
    api_docs: bool = False
    # Shabbat closing: Hebcal location; "auto" follows Hebcal, "closed" forces the closed screen
    # (for demos and tests). There is deliberately no "always open" setting.
    geonameid: int = 281184
    israel: bool = True
    timezone: str = "Asia/Jerusalem"
    closing: str = "auto"

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

    def password(self, role: str) -> str:
        value = read_secret(getattr(self, f"{role}_password"), getattr(self, f"{role}_password_file"))
        if not value or len(value) < 10 or value.lower() in _PLACEHOLDERS:
            raise RuntimeError(f"CHAVRUTA_{role.upper()}_PASSWORD (or _FILE) must be set (10+ characters)")
        return value
