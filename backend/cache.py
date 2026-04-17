import json
import os
from typing import Any, Optional

import redis


class RedisCache:
    def __init__(self):
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.key_prefix = os.getenv("REDIS_KEY_PREFIX", "superhermes")
        self.default_ttl = int(os.getenv("REDIS_CACHE_TTL_SECONDS", "300"))
        self._client = None

    def _get_client(self):
        if self._client is None:
            self._client = redis.Redis.from_url(self.redis_url, decode_responses=True)
        return self._client

    def _key(self, key: str) -> str:
        return f"{self.key_prefix}:{key}"

    def get_json(self, key: str) -> Optional[Any]:
        try:
            value = self._get_client().get(self._key(key))
            if not value:
                return None
            return json.loads(value)
        except Exception:
            return None

    def get_many_json(self, keys: list[str]) -> dict[str, Any]:
        if not keys:
            return {}
        try:
            client = self._get_client()
            values = client.mget([self._key(key) for key in keys])
            result = {}
            for key, value in zip(keys, values):
                if value:
                    result[key] = json.loads(value)
            return result
        except Exception:
            return {}

    def set_json(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        try:
            payload = json.dumps(value, ensure_ascii=False)
            self._get_client().setex(self._key(key), ttl or self.default_ttl, payload)
        except Exception:
            return

    def set_many_json(self, mapping: dict[str, Any], ttl: Optional[int] = None) -> None:
        if not mapping:
            return
        try:
            expires_in = ttl or self.default_ttl
            pipe = self._get_client().pipeline(transaction=False)
            for key, value in mapping.items():
                payload = json.dumps(value, ensure_ascii=False)
                pipe.setex(self._key(key), expires_in, payload)
            pipe.execute()
        except Exception:
            return

    def delete(self, key: str) -> None:
        try:
            self._get_client().delete(self._key(key))
        except Exception:
            return

    def delete_many(self, keys: list[str]) -> None:
        if not keys:
            return
        try:
            self._get_client().delete(*[self._key(key) for key in keys])
        except Exception:
            return

    def delete_pattern(self, pattern: str) -> None:
        try:
            full_pattern = self._key(pattern)
            keys = self._get_client().keys(full_pattern)
            if keys:
                self._get_client().delete(*keys)
        except Exception:
            return


cache = RedisCache()
