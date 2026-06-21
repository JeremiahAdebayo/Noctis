from typing import Dict, Any


class Cache:
    def __init__(self):
        self._store: Dict[str, Any] = {}

    def get(self, key: str):
        return self._store.get(key)

    def set(self, key: str, value: Any):
        self._store[key] = value

    def clear(self):
        self._store.clear()