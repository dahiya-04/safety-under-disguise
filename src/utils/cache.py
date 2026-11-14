"""Thread-safe append-only JSONL key/value cache.

Used for model responses so interrupted or repeated runs never pay for the
same API call twice. Each line is ``{"key": ..., "value": ...}``; later lines
override earlier ones, and a partially written last line is ignored.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any


def stable_hash(*parts: Any) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class JSONLCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._data: dict[str, Any] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(line)
                    self._data[rec["key"]] = rec["value"]
                except (json.JSONDecodeError, KeyError):
                    continue  # tolerate a torn final line after a crash

    def get(self, key: str) -> Any | None:
        with self._lock:
            return self._data.get(key)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")
