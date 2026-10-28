"""Memory 2.0, Phase 8.2c-1: the snapshot manifest. A pure value layer over `MemoryStore.snapshot()`.

It builds, serialises, parses and checks a manifest. It reads no file, takes no lock, writes nothing and contains no
digest formula of its own: every digest is computed by `memory_store.manifest_digest` / `memory_store.snapshot_digest`,
the one authoritative definitions. It is connected to nothing in Jarvis yet.

MANIFEST (`memory-manifest/1`), canonical JSON: sorted keys, compact separators, ASCII-only, one line, no trailing newline:

    {"count": N, "files": [[name, sha256], ...], "nonces": [[hash32, memory_id, confirmed_at], ...],
     "schema": "memory-manifest/1", "snapshot_digest": "<64 hex>", "state_digest": "<64 hex>"}

`files` is a LIST OF PAIRS in ascending name order, not an object: a JSON object with a repeated key is silently
last-wins when parsed, which is the wrong failure mode for an integrity record. An empty snapshot is `count` 0 with
empty `files` and `nonces`. Parsing refuses anything that is not exactly the canonical text of a consistent manifest.

WHAT THIS IS NOT. These are integrity checks, not authentication: nothing here shows who produced a manifest, and any
code running as the same user can build a manifest and every digest in it. A manifest also records an observation made
at one moment; it says nothing about the folder afterwards.
"""
import json
from dataclasses import dataclass

from backtalk import memory_store as S

SCHEMA = "memory-manifest/1"
_KEYS = frozenset({"count", "files", "nonces", "schema", "snapshot_digest", "state_digest"})


class ManifestError(Exception):
    """A refusal. `reason` is a short code; `details` are codes only, never content."""

    def __init__(self, reason, details=()):
        super().__init__(reason + (": " + ", ".join(map(str, details)) if details else ""))
        self.reason, self.details = reason, tuple(details)


@dataclass(frozen=True)
class Manifest:
    files: tuple                  # ((name, sha256_hex), ...) ascending by name
    nonces: tuple                 # ((hash32, memory_id, confirmed_at), ...) ascending
    state_digest: str
    snapshot_digest: str

    @property
    def count(self) -> int:
        return len(self.files)


def _from_parts(files, nonces) -> Manifest:
    try:
        entries = S.normalized_entries(files)
        rows = S.normalized_nonce_rows(nonces)
        return Manifest(tuple((n, h) for n, h in entries), tuple(tuple(r) for r in rows),
                        S.manifest_digest(entries), S.snapshot_digest(entries, rows))
    except (TypeError, ValueError) as e:
        raise ManifestError("bad_manifest", (str(e),)) from None


def build(snap) -> Manifest:
    """The manifest of a successful snapshot. Digests are recomputed here, never copied from the snapshot."""
    if not isinstance(snap, S.Snapshot) or not snap.ok:
        raise ManifestError("snapshot_not_ok", (getattr(snap, "reason", "not_a_snapshot"),))
    return _from_parts(snap.files, snap.nonces)


def to_dict(m: Manifest) -> dict:
    return {"schema": SCHEMA, "count": m.count, "files": [[n, h] for n, h in m.files],
            "nonces": [list(r) for r in m.nonces], "state_digest": m.state_digest,
            "snapshot_digest": m.snapshot_digest}


def canonical_text(m: Manifest) -> str:
    return S.canonical_text(to_dict(m))


def _no_duplicate_keys(pairs):
    keys = [k for k, _ in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate_key")
    return dict(pairs)


def _refuse_constant(_name):
    raise ValueError("non_finite")


def parse(text) -> Manifest:
    """The manifest in `text`, or ManifestError. Strict: exact keys, known schema, consistent count and digests, and the
    text must be byte-for-byte the canonical form (no extra whitespace, other key order or \\u escapes)."""
    if not isinstance(text, str):
        raise ManifestError("bad_manifest", ("type",))
    try:
        data = json.loads(text, object_pairs_hook=_no_duplicate_keys, parse_constant=_refuse_constant)
    except ValueError:
        raise ManifestError("bad_manifest", ("json",)) from None
    if not isinstance(data, dict) or set(data) != _KEYS:
        raise ManifestError("bad_manifest", ("keys",))
    if data["schema"] != SCHEMA:
        raise ManifestError("bad_manifest", ("schema",))
    files, nonces = data["files"], data["nonces"]
    if not (isinstance(files, list) and all(isinstance(e, list) and len(e) == 2 for e in files)
            and isinstance(nonces, list) and all(isinstance(r, list) and len(r) == 3 for r in nonces)):
        raise ManifestError("bad_manifest", ("shape",))
    m = _from_parts([tuple(e) for e in files], nonces)
    if type(data["count"]) is not int or data["count"] != m.count:
        raise ManifestError("bad_manifest", ("count",))
    if data["state_digest"] != m.state_digest or data["snapshot_digest"] != m.snapshot_digest:
        raise ManifestError("bad_manifest", ("digest",))
    if canonical_text(m) != text:
        raise ManifestError("bad_manifest", ("not_canonical",))
    return m


def validate_against(m: Manifest, snap) -> None:
    """Raise ManifestError unless `m` describes exactly this successful snapshot. Every field is compared with values
    recomputed from the snapshot's files and nonces; a digest is never trusted because it is present."""
    if not isinstance(m, Manifest):
        raise ManifestError("bad_manifest", ("type",))
    if not isinstance(snap, S.Snapshot) or not snap.ok:
        raise ManifestError("snapshot_not_ok", (getattr(snap, "reason", "not_a_snapshot"),))
    expected = _from_parts(snap.files, snap.nonces)
    for field in ("files", "nonces", "state_digest", "snapshot_digest"):
        if getattr(m, field) != getattr(expected, field):
            raise ManifestError("manifest_mismatch", (field,))
    if (snap.state_digest, snap.snapshot_digest) != (expected.state_digest, expected.snapshot_digest):
        raise ManifestError("manifest_mismatch", ("snapshot_fields",))
