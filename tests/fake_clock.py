"""TEST-ONLY deterministic clock for MemoryStore(clock=...). Never imported by production code (enforced by an AST test)."""
from datetime import datetime, timedelta, timezone

from backtalk import memory as M


class FakeClock:
    def __init__(self, start="2026-02-01T00:00:00Z"):
        self.t = M.parse_ts(start)

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t = self.t + timedelta(seconds=seconds)

    def stamp(self, offset=0):
        return (self.t + timedelta(seconds=offset)).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
