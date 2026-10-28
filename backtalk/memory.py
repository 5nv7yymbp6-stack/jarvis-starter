"""The pure semantic core of Memory 2.0 (Phase 1): what a memory item IS, when it is valid, which item is the
current truth for a logical fact, and when two items are the same fact.

PURE. This module reads no file, opens no socket, calls no model, reads no clock (every time is passed in by the
caller), and imports nothing of Jarvis. It changes nothing it is handed: items are frozen, collections are copied,
results are new tuples. Nothing in the running system imports it yet.

THE ITEM. One atomic statement about one logical fact:
    subject + attribute   the logical key ("user.residence"): the SAME fact across time
    value                 what is claimed (needed: without it two items for one key cannot be told apart)
    type                  identity | preference | fact | goal | event | instruction
    confidence            stated (the user said it) | inferred (Jarvis concluded it)
    valid_from / valid_to when it is true: [valid_from, valid_to), valid_to None = still true
    status                active | superseded | archived
    supersedes            ids of the items this one replaces (an explicit, checkable link)
    sensitivity           normal | private | never-expose
    source                where it came from: kind + reference
    created / updated     when the record was made / last changed
Timestamps are UTC ISO strings, exactly `YYYY-MM-DDTHH:MM:SSZ`.

FAIL CLOSED. An item that does not validate is rejected with deterministic error codes; nothing is coerced,
trimmed or repaired. A time the caller supplies that is not a time is a ValueError, not a guess.

SUPERSESSION (resolve). For one key at time T, among items valid at T:
    * an explicit `supersedes` link beats any ordering (the superseded item is history);
    * a stated item outranks an inferred one: an inferred item never overrides a stated one, however new;
    * among equal authority, the newer wins: later `valid_from`, then later `created`;
    * equal in both and different in value is a CONFLICT, returned explicitly, with no active item guessed;
    * equal in both and the same value is a duplicate (the lowest id stands);
    * archived items are never active and never contested.
"""
import hashlib
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone

TYPES = ("identity", "preference", "fact", "goal", "event", "instruction")
STATUSES = ("active", "superseded", "archived")
CONFIDENCES = ("stated", "inferred")
SENSITIVITIES = ("normal", "private", "never-expose")
SOURCE_KINDS = ("user", "session", "note", "dream", "import")

_AUTHORITY = {"stated": 2, "inferred": 1}
_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z")
_TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
_ID_RE = re.compile(r"^mem_[0-9a-f]{12}\Z")
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}\Z")
VALUE_MAX = 2000
REF_MAX = 300
SUPERSEDES_MAX = 8

REQUIRED_FIELDS = ("id", "type", "subject", "attribute", "value", "created", "updated", "source",
                   "confidence", "valid_from", "status", "sensitivity")
OPTIONAL_FIELDS = ("valid_to", "supersedes")

EXACT_DUPLICATE, SAME_KEY_DIFFERENT_VALUE, DISTINCT = "exact_duplicate", "same_key_different_value", "distinct"


@dataclass(frozen=True)
class Source:
    kind: str
    ref: str


@dataclass(frozen=True)
class MemoryItem:
    id: str
    type: str
    subject: str
    attribute: str
    value: str
    created: str
    updated: str
    source: Source
    confidence: str
    valid_from: str
    status: str
    sensitivity: str
    valid_to: object = None            # a timestamp string, or None for "still true"
    supersedes: tuple = ()             # ids of the items this one replaces


@dataclass(frozen=True)
class Conflict:
    key: str
    candidates: tuple                  # equally authoritative, equally recent, different values; sorted by id


@dataclass(frozen=True)
class Resolution:
    active: tuple                      # the current item of each key, sorted by key
    conflicts: tuple                   # Conflict, sorted by key; no active item is chosen for these keys
    history: tuple                     # (item, reason), sorted by (key, valid_from, created, id)
    rejected: tuple                    # (id or "#index", error codes), in input order


# ---- time ----------------------------------------------------------------------------------------

def parse_ts(s):
    """A UTC datetime for a strictly formatted timestamp string, else None."""
    if not isinstance(s, str) or not _TS_RE.match(s):
        return None
    try:
        return datetime.strptime(s, _TS_FORMAT).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _at(at):
    """The caller's time as an aware UTC datetime. Anything else is a ValueError: no wall clock is ever read."""
    if isinstance(at, datetime):
        if at.tzinfo is None or at.utcoffset() is None:
            raise ValueError("time must be timezone-aware")
        return at.astimezone(timezone.utc)
    t = parse_ts(at)
    if t is None:
        raise ValueError("time must be YYYY-MM-DDTHH:MM:SSZ or an aware datetime")
    return t


# ---- keys, values, identity ----------------------------------------------------------------------

def key_of(item) -> str:
    """The logical key: the same fact across time (names cannot contain '.', so this is unambiguous)."""
    return f"{item.subject}.{item.attribute}"


def normalize_value(value) -> str:
    """A value as compared for duplicates: Unicode-normalized, case-folded, whitespace collapsed."""
    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def fingerprint(value) -> str:
    return hashlib.sha256(normalize_value(value).encode("utf-8")).hexdigest()[:16]


def derive_id(subject, attribute, value, created) -> str:
    """A stable id from what the item says and when it was made."""
    raw = "\x1f".join((str(subject), str(attribute), normalize_value(value), str(created)))
    return "mem_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


# ---- validation ----------------------------------------------------------------------------------

def _blank(v) -> bool:
    return v is None or (isinstance(v, str) and not v.strip())


def validate(item) -> tuple:
    """Deterministic error codes for an item, or () if it is valid. Never raises."""
    if not isinstance(item, MemoryItem):
        return ("invalid:not_a_memory_item",)
    e = []
    if _blank(item.id):
        e.append("missing:id")
    elif not (isinstance(item.id, str) and _ID_RE.match(item.id)):
        e.append("invalid:id")
    if _blank(item.type):
        e.append("missing:type")
    elif item.type not in TYPES:
        e.append("invalid:type")
    for name in ("subject", "attribute"):
        v = getattr(item, name)
        if _blank(v):
            e.append(f"missing:{name}")
        elif not (isinstance(v, str) and _NAME_RE.match(v)):
            e.append(f"invalid:{name}")
    if _blank(item.value):
        e.append("missing:value")
    elif not isinstance(item.value, str) or len(item.value) > VALUE_MAX:
        e.append("invalid:value")
    s = item.source
    if s is None:
        e.append("missing:source")
    elif not (isinstance(s, Source) and s.kind in SOURCE_KINDS and isinstance(s.ref, str) and s.ref.strip()
              and len(s.ref) <= REF_MAX and "\n" not in s.ref and "\r" not in s.ref):
        e.append("invalid:source")
    for name, allowed in (("confidence", CONFIDENCES), ("status", STATUSES), ("sensitivity", SENSITIVITIES)):
        v = getattr(item, name)
        if _blank(v):
            e.append(f"missing:{name}")
        elif v not in allowed:
            e.append(f"invalid:{name}")
    times = {}
    for name in ("created", "updated", "valid_from"):
        v = getattr(item, name)
        if _blank(v):
            e.append(f"missing:{name}")
        elif parse_ts(v) is None:
            e.append(f"invalid:timestamp:{name}")
        else:
            times[name] = parse_ts(v)
    if item.valid_to is not None:
        if parse_ts(item.valid_to) is None:
            e.append("invalid:timestamp:valid_to")
        else:
            times["valid_to"] = parse_ts(item.valid_to)
    if "created" in times and "updated" in times and times["updated"] < times["created"]:
        e.append("order:updated_before_created")
    if "valid_from" in times and "valid_to" in times and times["valid_to"] <= times["valid_from"]:
        e.append("range:valid_to_not_after_valid_from")
    if item.status == "superseded" and item.valid_to is None:
        e.append("status:superseded_without_valid_to")
    sup = item.supersedes
    if not isinstance(sup, tuple) or len(sup) > SUPERSEDES_MAX \
            or not all(isinstance(x, str) and _ID_RE.match(x) for x in sup) or len(set(sup)) != len(sup):
        e.append("invalid:supersedes")
    elif item.id in sup:
        e.append("supersedes:self")
    return tuple(e)


def parse(data):
    """(MemoryItem, ()) for a valid plain mapping, else (None, errors). Strings stay strings: nothing is coerced,
    trimmed or defaulted, an unknown field is an error, and `valid_to` / `supersedes` may be absent."""
    if not hasattr(data, "keys") or not hasattr(data, "get"):
        return None, ("invalid:not_a_mapping",)
    keys = list(data.keys())
    errors = [f"unknown_field:{k}" for k in sorted(map(str, keys)) if k not in REQUIRED_FIELDS + OPTIONAL_FIELDS]
    errors += [f"missing:{k}" for k in REQUIRED_FIELDS if k not in keys]
    for k in REQUIRED_FIELDS:
        if k in keys and k != "source" and not isinstance(data[k], str):
            errors.append(f"invalid:{k}")
    src = data.get("source")
    source = None
    if "source" in keys:
        if hasattr(src, "keys") and sorted(map(str, src.keys())) == ["kind", "ref"] \
                and isinstance(src.get("kind"), str) and isinstance(src.get("ref"), str):
            source = Source(src["kind"], src["ref"])
        else:
            errors.append("invalid:source")
    valid_to = data.get("valid_to")
    if valid_to is not None and not isinstance(valid_to, str):
        errors.append("invalid:valid_to")
    sup = data.get("supersedes", ())
    if isinstance(sup, (list, tuple)) and all(isinstance(x, str) for x in sup):
        sup = tuple(sup)
    else:
        errors.append("invalid:supersedes")
        sup = ()
    if errors:
        return None, tuple(dict.fromkeys(errors))
    item = MemoryItem(**{k: data[k] for k in REQUIRED_FIELDS if k != "source"}, source=source,
                      valid_to=valid_to, supersedes=sup)
    errs = validate(item)
    return (None, errs) if errs else (item, ())


def to_dict(item) -> dict:
    """The plain, serializable form of an item (the inverse of parse)."""
    return {"id": item.id, "type": item.type, "subject": item.subject, "attribute": item.attribute,
            "value": item.value, "created": item.created, "updated": item.updated,
            "source": {"kind": item.source.kind, "ref": item.source.ref}, "confidence": item.confidence,
            "valid_from": item.valid_from, "valid_to": item.valid_to, "status": item.status,
            "supersedes": list(item.supersedes), "sensitivity": item.sensitivity}


# ---- duplicates ----------------------------------------------------------------------------------

def classify_pair(a, b) -> str:
    """exact_duplicate (same key, same type, same normalized value), same_key_different_value, or distinct.
    A same-key pair of different types is never an exact duplicate."""
    if key_of(a) != key_of(b):
        return DISTINCT
    if a.type == b.type and normalize_value(a.value) == normalize_value(b.value):
        return EXACT_DUPLICATE
    return SAME_KEY_DIFFERENT_VALUE


def find_duplicates(candidate, existing) -> tuple:
    """(id, relation) for every existing item that is an exact duplicate of the candidate or says something
    different about the same key. Distinct items and the candidate itself are left out. Deterministic order."""
    order = {EXACT_DUPLICATE: 0, SAME_KEY_DIFFERENT_VALUE: 1}
    hits = [(x.id, classify_pair(candidate, x)) for x in existing if x.id != candidate.id]
    return tuple(sorted(((i, r) for i, r in hits if r != DISTINCT), key=lambda h: (order[h[1]], h[0])))


# ---- when an item is true ------------------------------------------------------------------------

def is_active(item, at, *, include_archived=False) -> bool:
    """Is the item true at time `at`? valid_from <= at < valid_to (valid_to None = open). An invalid item is
    never active, and an archived one only when asked for (a history view). A superseded item is true for the
    time before it ended, which is what makes a historical question answerable."""
    t = _at(at)
    if validate(item):
        return False
    if item.status == "archived" and not include_archived:
        return False
    if parse_ts(item.valid_from) > t:
        return False
    return item.valid_to is None or t < parse_ts(item.valid_to)


# ---- the current truth ---------------------------------------------------------------------------

def _order(item):
    return parse_ts(item.valid_from), parse_ts(item.created)


def resolve(items, at) -> Resolution:
    """For each logical key, the item that is true at `at`, or an explicit Conflict. Pure and deterministic:
    active, conflicts and history are the same in any input order (`rejected` lists in input order), and
    nothing in `items` is changed."""
    t = _at(at)
    items = tuple(items)
    rejected = {}                                           # index -> errors
    shaped = {}
    for i, it in enumerate(items):
        errs = validate(it)
        if errs:
            rejected[i] = errs
        else:
            shaped[i] = it
    ids = Counter(it.id for it in shaped.values())
    for i, it in list(shaped.items()):
        if ids[it.id] > 1:
            rejected[i] = ("id:duplicate",)
            del shaped[i]
    while True:                                             # a rejected item can strand another's link
        by_id = {it.id: it for it in shaped.values()}
        bad = {}
        for i, it in shaped.items():                        # judged against one snapshot: input order cannot matter
            errs = []
            for ref in it.supersedes:
                old = by_id.get(ref)
                if old is None:
                    errs.append(f"supersedes:unknown:{ref}")
                elif key_of(old) != key_of(it):
                    errs.append(f"supersedes:key_mismatch:{ref}")
                elif _AUTHORITY[it.confidence] < _AUTHORITY[old.confidence]:
                    errs.append(f"supersedes:authority:{ref}")
                elif _reaches(by_id, ref, it.id):
                    errs.append(f"supersedes:cycle:{ref}")
            if errs:
                bad[i] = tuple(errs)
        if not bad:
            break
        for i, errs in bad.items():
            rejected[i] = errs
            del shaped[i]
    live = sorted(shaped.values(), key=lambda x: x.id)
    reasons = {}                                            # id -> reason it is not the current truth
    candidates = {}                                         # key -> items true at t
    for it in live:
        if it.status == "archived":
            reasons[it.id] = "archived"
        elif parse_ts(it.valid_from) > t:
            reasons[it.id] = "not_yet_valid"
        elif it.valid_to is not None and parse_ts(it.valid_to) <= t:
            reasons[it.id] = "ended"
        else:
            candidates.setdefault(key_of(it), []).append(it)
    active, conflicts = [], []
    for key in sorted(candidates):
        group = candidates[key]
        replaced = {ref for it in group for ref in it.supersedes}
        for it in group:
            if it.id in replaced:
                reasons[it.id] = "superseded"
        group = [it for it in group if it.id not in replaced]
        top = max(_AUTHORITY[it.confidence] for it in group)
        group, lower = ([it for it in group if _AUTHORITY[it.confidence] == top],
                        [it for it in group if _AUTHORITY[it.confidence] < top])
        newest = max(_order(it) for it in group)
        for it in lower:                                     # weaker authority: replaced if older, shadowed if not
            reasons[it.id] = "superseded" if _order(it) < newest else "shadowed"
        for it in group:
            if _order(it) < newest:
                reasons[it.id] = "superseded"
        leaders = sorted((it for it in group if _order(it) == newest), key=lambda x: x.id)
        if len({(it.type, normalize_value(it.value)) for it in leaders}) > 1:
            conflicts.append(Conflict(key, tuple(leaders)))
            continue
        active.append(leaders[0])
        for it in leaders[1:]:
            reasons[it.id] = "duplicate"
    by_id = {it.id: it for it in live}
    history = sorted(((by_id[i], r) for i, r in reasons.items()),
                     key=lambda h: (key_of(h[0]), h[0].valid_from, h[0].created, h[0].id))
    return Resolution(active=tuple(active), conflicts=tuple(conflicts), history=tuple(history),
                      rejected=tuple((items[i].id if isinstance(getattr(items[i], "id", None), str) else f"#{i}",
                                      rejected[i]) for i in sorted(rejected)))


def _reaches(by_id, start, target) -> bool:
    """Does following `supersedes` links from `start` reach `target`?"""
    seen, stack = set(), [start]
    while stack:
        cur = stack.pop()
        if cur == target:
            return True
        if cur in seen or cur not in by_id:
            continue
        seen.add(cur)
        stack.extend(by_id[cur].supersedes)
    return False
