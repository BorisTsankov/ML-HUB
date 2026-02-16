import joblib
from typing import Any

class ModelStore:
    def __init__(self):
        self._cache: dict[str, Any] = {}

    def load_bundle(self, path: str):
        if path not in self._cache:
            self._cache[path] = joblib.load(path)
        return self._cache[path]

model_store = ModelStore()
