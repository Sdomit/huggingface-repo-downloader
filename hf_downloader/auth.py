from __future__ import annotations

from dataclasses import dataclass

import keyring
from huggingface_hub import get_token
from keyring.errors import KeyringError

SERVICE_NAME = "HF Repo Downloader"
TOKEN_NAME = "huggingface-token"


@dataclass(slots=True)
class ResolvedToken:
    token: str | bool | None
    source: str


class TokenStore:
    def get_saved_token(self) -> str | None:
        try:
            return keyring.get_password(SERVICE_NAME, TOKEN_NAME)
        except KeyringError:
            return None

    def save_token(self, token: str) -> None:
        keyring.set_password(SERVICE_NAME, TOKEN_NAME, token)

    def clear_token(self) -> None:
        try:
            keyring.delete_password(SERVICE_NAME, TOKEN_NAME)
        except KeyringError:
            return


class AuthResolver:
    def __init__(self, token_store: TokenStore | None = None) -> None:
        self.token_store = token_store or TokenStore()

    def resolve(self, session_token: str | None = None) -> ResolvedToken:
        cleaned_session = (session_token or "").strip()
        if cleaned_session:
            return ResolvedToken(token=cleaned_session, source="session")

        saved = self.token_store.get_saved_token()
        if saved:
            return ResolvedToken(token=saved, source="keyring")

        existing = get_token()
        if existing:
            return ResolvedToken(token=existing, source="huggingface")

        return ResolvedToken(token=None, source="anonymous")

