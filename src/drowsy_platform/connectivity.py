from __future__ import annotations

import time
import urllib.error
import urllib.request


class NetworkStatusProbe:
    def __init__(self, probe_url: str, timeout_seconds: float, ttl_seconds: int) -> None:
        self.probe_url = probe_url
        self.timeout_seconds = timeout_seconds
        self.ttl_seconds = ttl_seconds
        self._last_status = False
        self._last_checked_at = 0.0

    def is_online(self) -> bool:
        now = time.monotonic()
        if now - self._last_checked_at < self.ttl_seconds:
            return self._last_status

        try:
            with urllib.request.urlopen(self.probe_url, timeout=self.timeout_seconds):
                self._last_status = True
        except (urllib.error.URLError, TimeoutError, ValueError):
            self._last_status = False

        self._last_checked_at = now
        return self._last_status
