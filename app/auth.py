import os
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

# Header adı: X-API-Key
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_valid_keys() -> set:
    raw = os.getenv("API_KEYS", "")
    return {k.strip() for k in raw.split(",") if k.strip()}


def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    """
    Her korumalı endpoint için dependency olarak kullanılır.
    Header'da geçerli bir X-API-Key yoksa 401 döner.
    """
    valid_keys = get_valid_keys()

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-API-Key header missing",
        )

    if api_key not in valid_keys:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )

    return api_key
