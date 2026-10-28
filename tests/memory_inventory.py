"""READ-ONLY inventory of the Obsidian vault against the Memory 2.0 schema (Phase 1).

It answers: how much of the existing vault already fits the Memory 2.0 model, and where would the two collide.
It opens files for reading only, writes nothing, calls nothing, and migrates nothing. It never prints a secret:
for secret-like material it reports the file, the line number and the CATEGORY, never the text.

    python -m tests.memory_inventory [/path/to/vault]

Not named test_*.py on purpose: unittest must not run it against a vault that changes every day. The unit tests
exercise it on a synthetic vault (tests/test_memory.py).
"""
import os
import re
import sys
from collections import Counter, defaultdict

from backtalk import memory as M

DEFAULT_VAULT = "/Users/yousif/Desktop/jarvis/jarvis"
SKIP_DIRS = {".git", ".obsidian", ".trash"}

# A vault note is a DOCUMENT; a Memory 2.0 item is one atomic statement. Only these vault note types have a
# single natural item type, and even then a note is a SOURCE of items, not an item.
NOTE_TYPE_TO_MEMORY = {"log": "event", "plan": "goal", "protocol": "instruction"}

# frontmatter keys the vault already uses that Memory 2.0 also wants, and what each would mean
SHARED_KEYS = ("status", "type", "source", "created")
# keys only Memory 2.0 uses; seeing one in the vault would be a collision with a note that is not an item
MEMORY_ONLY_KEYS = ("id", "subject", "attribute", "value", "confidence", "valid_from", "valid_to",
                    "supersedes", "sensitivity", "updated")

_SECRET_PATTERNS = (
    ("assignment:credential-name", re.compile(r"(?i)\b(api[_ -]?key|secret|token|password|passwd)\b\s*[:=]\s*\S{8,}")),
    ("bearer-token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]{16,}")),
    ("provider-key:sk-", re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{16,}")),
    ("github-token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}")),
    ("aws-access-key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("private-key-block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
)


def _notes(vault):
    for root, dirs, files in os.walk(vault):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for f in sorted(files):
            if f.endswith(".md"):
                yield os.path.join(root, f)


def _snapshot(vault):
    out = {}
    for p in _notes(vault):
        st = os.stat(p)
        out[os.path.relpath(p, vault)] = (st.st_size, st.st_mtime_ns)
    return out


def _frontmatter(text):
    """{key: first-line value} of a leading --- block, or None if the note has none."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    out = {}
    for ln in lines[1:]:
        if ln.strip() == "---":
            return out
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_-]*):\s*(.*)$", ln)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return None                                  # never closed: not frontmatter


def inventory(vault=DEFAULT_VAULT) -> dict:
    before = _snapshot(vault)
    r = {"total": 0, "required_frontmatter": 0, "no_frontmatter": [], "missing_required": [], "created": 0,
         "created_iso_date_only": 0, "created_iso_timestamp": 0, "source_family": 0,
         "note_types": Counter(), "mappable": Counter(), "documents": Counter(), "unmappable": 0,
         "status_values": Counter(), "status_outside_memory": 0, "type_overlap_with_memory": 0,
         "memory_only_keys": [], "secret_like": [], "duplicate_basenames": {}, "folder_number_collisions": {}}
    basenames = defaultdict(list)
    for path in _notes(vault):
        rel = os.path.relpath(path, vault)
        with open(path, "r", encoding="utf-8", errors="replace") as f:          # read-only
            text = f.read()
        r["total"] += 1
        basenames[os.path.basename(rel).lower()].append(rel)
        fm = _frontmatter(text)
        for n, ln in enumerate(text.splitlines(), 1):
            for cat, pat in _SECRET_PATTERNS:
                if pat.search(ln):
                    r["secret_like"].append((rel, n, cat))
                    break
        if fm is None:
            r["no_frontmatter"].append(rel)
            r["missing_required"].append((rel, ("status", "project", "type")))
            r["unmappable"] += 1
            continue
        missing = tuple(k for k in ("status", "project", "type") if not fm.get(k))
        if missing:
            r["missing_required"].append((rel, missing))
        else:
            r["required_frontmatter"] += 1
        if fm.get("created"):
            r["created"] += 1
            c = fm["created"]
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", c):
                r["created_iso_date_only"] += 1
            elif M.parse_ts(c):
                r["created_iso_timestamp"] += 1
        if any(fm.get(k) for k in ("source", "creator", "pulled")):
            r["source_family"] += 1
        t = fm.get("type")
        r["note_types"][t] += 1
        r["status_values"][fm.get("status")] += 1
        if fm.get("status") and fm["status"] not in M.STATUSES:
            r["status_outside_memory"] += 1
        if t in M.TYPES:
            r["type_overlap_with_memory"] += 1
        if t in NOTE_TYPE_TO_MEMORY:
            r["mappable"][NOTE_TYPE_TO_MEMORY[t]] += 1
        else:
            r["documents"][t] += 1
            r["unmappable"] += 1
        hit = tuple(k for k in MEMORY_ONLY_KEYS if k in fm)
        if hit:
            r["memory_only_keys"].append((rel, hit))
    r["duplicate_basenames"] = {b: sorted(v) for b, v in basenames.items() if len(v) > 1}
    prefixes = defaultdict(list)
    for d in sorted(os.listdir(vault)):
        m = re.match(r"^(\d+)\s*-\s", d)
        if m and os.path.isdir(os.path.join(vault, d)):
            prefixes[m.group(1)].append(d)
    r["folder_number_collisions"] = {k: v for k, v in sorted(prefixes.items()) if len(v) > 1}
    r["vault_unchanged_during_inventory"] = (before == _snapshot(vault))
    return r


def render(r) -> str:
    L = []
    add = L.append
    add(f"Memory 2.0 vault inventory (read-only)  -  {r['total']} notes")
    add(f"  status/project/type all present : {r['required_frontmatter']}")
    add(f"  missing one or more of them     : {len(r['missing_required'])}"
        + (f"  {[m[0] for m in r['missing_required']]}" if r['missing_required'] else ""))
    add(f"  with `created`                  : {r['created']}  (date-only {r['created_iso_date_only']}, "
        f"full ISO timestamp {r['created_iso_timestamp']}; Memory 2.0 needs a timestamp)")
    add(f"  with source/creator/pulled      : {r['source_family']}  (free text; Memory 2.0 `source` is kind+ref)")
    add(f"  note types                      : {dict(r['note_types'])}")
    add(f"  cleanly mappable to one item type: {sum(r['mappable'].values())}  {dict(r['mappable'])}")
    add(f"  documents with no single item type: {sum(r['documents'].values())}  {dict(r['documents'])}")
    add(f"  cannot be mapped                : {r['unmappable']}")
    add("  collisions with the Memory 2.0 schema:")
    add(f"    `status`: vault values {dict(r['status_values'])}; Memory 2.0 allows {list(M.STATUSES)}; "
        f"{r['status_outside_memory']} notes use a value Memory 2.0 does not")
    add(f"    `type`  : {r['type_overlap_with_memory']} notes use a vault type that is also a Memory 2.0 type "
        f"(vault types are document kinds)")
    add(f"    `source`/`created`: used with a different shape (see above)")
    add(f"    notes already carrying Memory-2.0-only keys: {r['memory_only_keys'] or 'none'}")
    add(f"    duplicate basenames across folders: {r['duplicate_basenames'] or 'none'}")
    add(f"    folder-number collisions: {r['folder_number_collisions'] or 'none'}")
    add(f"  secret-like material (file:line category; values never printed): {len(r['secret_like'])}")
    for rel, n, cat in r["secret_like"]:
        add(f"    {rel}:{n}  {cat}")
    add(f"  vault unchanged during the inventory: {r['vault_unchanged_during_inventory']}")
    return "\n".join(L)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    vault = argv[0] if argv else DEFAULT_VAULT
    print(render(inventory(vault)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
