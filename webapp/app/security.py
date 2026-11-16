"""Autenticacao via Bearer Token em tempo constante para rotas de integracao."""

from __future__ import annotations

import secrets
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import Settings, get_settings


bearer_scheme = HTTPBearer(auto_error=False)


def require_bearer_token(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    settings: Settings = Depends(get_settings),
) -> str:
    """Exige Bearer Token valido quando ARBORGRID_REQUIRE_AUTH=true."""
    if not settings.require_auth:
        return "auth-disabled-local"

    expected = settings.api_token
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ARBORGRID_API_TOKEN nao configurado no servidor.",
        )

    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credenciais Bearer ausentes ou malformadas.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not secrets.compare_digest(credentials.credentials, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token de acesso invalido.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return "authenticated"
