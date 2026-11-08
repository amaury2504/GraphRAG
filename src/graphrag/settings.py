from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr
    postgres_dsn: SecretStr
    # Empty or missing: /ask answers 503 and everything else keeps working.
    mistral_api_key: SecretStr | None = None
    mistral_model: str = "mistral-small-latest"
    # Which model writes the answers. "claude" runs the local `claude -p` command (personal
    # use, see adapters/claude_headless.py) and needs no Mistral key.
    llm_provider: Literal["mistral", "claude"] = "mistral"
    claude_model: str | None = None
    # A warning is logged once the language model has used more tokens than this.
    token_alert: int = 500_000
    # Load the embedding model and the language model client at startup instead of on the
    # first question (the first /ask is then fast, the start is slower).
    preload_models: bool = False
