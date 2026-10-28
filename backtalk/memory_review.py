"""The Memory 2.0 review state machine (Phase 7): turn the Phase 6 candidates into DECISIONS THE PERSON MADE, one item at a
time, recorded in a tamper-evident ledger. REVIEW ONLY: nothing here, or in the CLI that drives it, can write a memory.

PURE. No file, no socket, no model, no clock (the caller passes `now`), no state. It imports only the pure core, gate,
extractor and key registry, and it never imports the store. Nothing live imports it.

WHAT A DECISION IS. A decision is a ledger record, one per action, hash-chained (each record holds the hash of the one
before, so an edit, a reorder or a deletion is detectable). The actions:
    lock      the person ADOPTS the candidate exactly as shown. They must type the word `lock`; nothing else counts.
    edit      the person types their own value (and the sentence it rests on). Their words, re-screened.
    reject    with a reason code. Final.
    skip / defer   no decision; nothing is recorded as a fact; the item stays reviewable.
    stale     the source no longer says what it said; the item cannot be locked until it is re-extracted.
    conflict_choice   how the person resolved a conflict group. Never inferred, never defaulted.
Lock and edit are FINAL for that candidate: it cannot be reviewed twice.

PROVENANCE (decision D2). A locked note-written candidate means the person ADOPTED it in review. The record keeps the
candidate's ORIGINAL provenance untouched under `origin` (kind, reference, hash, tier: "note") and records the adoption
separately under `adoption` (by the user, at this instant, over the terminal). The note is never relabelled as the
user's own statement. The gate dry-run is run on the adoption (source user, stated), because that is the human event.

TIME (decision D3). `valid_from` is the confirmation instant. The only exception is a locked, unedited candidate whose own
evidence is a genuine transcript timestamp. A start date the person knows goes in the sentence they type, not in
`valid_from`: no historical validity is invented.

SAFETY. A candidate is offered only if its review_class is `offer`: private, instruction and template-default candidates
are deferred by policy (D5, D6). Every lock and edit re-runs the real gate in dry-run, re-screens the text for secrets,
injection and third-party contact details, recomputes sensitivity (it can go up, never down), validates the key against the
registry (collision-checked), and refuses a stale source or an unresolved conflict. An edit whose text turns out private is
recorded as blocked (`commit_blocked`) with its value kept out of the ledger: private values never enter the chain.
"""
import json
from collections import Counter
from dataclasses import dataclass

from backtalk import memory as M
from backtalk import memory_extract as X
from backtalk import memory_gate as G
from backtalk import memory_keys as K

LEDGER_FORMAT = "memory-review-ledger/1"
GENESIS = "0" * 64
REJECT_REASONS = ("not_true", "outdated", "not_mine", "too_private", "wrong_type")
LOCK_WORD = "lock"
FINAL_ACTIONS = ("lock", "edit", "reject")
OFFER = "offer"
BLOCK_PRIVATE = "private_requires_confirmed_option"


class ReviewError(Exception):
    """A refusal with a fixed code. Details are codes and ids only, never a value."""

    def __init__(self, code, details=()):
        super().__init__(code + (": " + ", ".join(map(str, details)) if details else ""))
        self.code, self.details = code, tuple(details)


@dataclass(frozen=True)
class Outcome:
    record: dict
    private: object = None          # {"seq", "value", "evidence"} for the 0600 side file, or None


# ---- the ledger ----------------------------------------------------------------------------------

def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def record_hash(rec) -> str:
    return X.sha(canonical_json({k: v for k, v in rec.items() if k != "hash"}))


def new_record(records, now, fields) -> dict:
    """The next record, chained to the last. `now` must be a strict UTC timestamp."""
    if M.parse_ts(now) is None:
        raise ValueError("now must be YYYY-MM-DDTHH:MM:SSZ")
    rec = {"seq": len(records), "prev": records[-1]["hash"] if records else GENESIS, "at": now}
    rec.update(fields)
    rec["hash"] = record_hash(rec)
    return rec


def verify_chain(records) -> list:
    """Problem codes for a ledger (empty = intact): a gap, a broken link, an altered record, a bad start."""
    problems = []
    prev = GENESIS
    for i, r in enumerate(records):
        if not isinstance(r, dict) or r.get("seq") != i:
            problems.append(f"seq:{i}")
        elif r.get("prev") != prev:
            problems.append(f"prev:{i}")
        elif r.get("hash") != record_hash(r):
            problems.append(f"hash:{i}")
        prev = r.get("hash") if isinstance(r, dict) else prev
    if records and records[0].get("action") != "open":
        problems.append("first_not_open")
    return problems


def open_record(records, now, *, run, pack_sha, pack_format) -> dict:
    if records:
        raise ReviewError("ledger_already_open")
    return new_record(records, now, {"action": "open", "run": run, "pack_sha": pack_sha, "pack_format": pack_format,
                                     "ledger_format": LEDGER_FORMAT, "registry_version": K.REGISTRY_VERSION})


def session_record(records, now, *, run, pack_sha) -> dict:
    """A new session on an existing ledger. It must be the SAME pack: a ledger is bound to one."""
    head = records[0] if records else {}
    if (head.get("run"), head.get("pack_sha")) != (run, pack_sha):
        raise ReviewError("ledger_mismatch")
    return new_record(records, now, {"action": "session", "run": run, "pack_sha": pack_sha})


# ---- the pack and the state ---------------------------------------------------------------------

_REQUIRED = ("id", "attribute", "type", "value", "evidence", "sensitivity", "review_class", "origin", "where", "line",
             "text_sha", "source_kind", "source_ref", "channel", "evidence_tier", "conflict_groups", "timestamp",
             "if_affirmed", "gate", "registry")


class Pack:
    """A validated Phase 6 pack: candidate views by id, and conflict groups by id."""

    def __init__(self, pack):
        if not isinstance(pack, dict) or pack.get("format") != X.PACK_FORMAT:
            raise ReviewError("bad_pack", ("format",))
        if pack.get("registry_version") != K.REGISTRY_VERSION:
            raise ReviewError("bad_pack", ("registry_version",))
        self.views, self.groups = {}, {}
        for c in pack.get("candidates", ()):
            if not isinstance(c, dict) or any(k not in c for k in _REQUIRED) or c["id"] in self.views:
                raise ReviewError("bad_pack", ("candidate",))
            self.views[c["id"]] = c
        for g in pack.get("conflicts", ()):
            if not isinstance(g, dict) or "id" not in g or g["id"] in self.groups:
                raise ReviewError("bad_pack", ("conflict",))
            self.groups[g["id"]] = g
        self.meta = pack.get("meta", {})

    def offered(self) -> list:
        return sorted(c for c, v in self.views.items() if v["review_class"] == OFFER)


class State:
    """What the ledger says now. Built only from a ledger whose chain verifies."""

    def __init__(self, records):
        problems = verify_chain(records)
        if problems:
            raise ReviewError("ledger_corrupt", problems)
        self.records = list(records)
        self.run = records[0]["run"] if records else None
        self.pack_sha = records[0]["pack_sha"] if records else None
        self.last, self.final, self.taken, self.choices = {}, {}, {}, {}
        for r in self.records:
            a = r.get("action")
            if a in ("lock", "edit", "reject", "skip", "defer", "stale"):
                self.last[r["candidate"]] = r
            if a in FINAL_ACTIONS:
                self.final[r["candidate"]] = r
            if a in ("lock", "edit"):
                self.taken[K.canonical(r["key"])] = r["candidate"]
            if a == "conflict_choice":
                self.choices[r["group"]] = r

    def status(self, cid) -> str:
        r = self.last.get(cid)
        if r is None:
            return "pending"
        return {"lock": "locked", "edit": "edited", "reject": "rejected", "skip": "skipped", "defer": "deferred",
                "stale": "stale"}[r["action"]]


def queue(pack, state) -> list:
    """Offered candidates still open, in review order: untouched first, uncontested before contested, then skipped or stale,
    deferred last. Final ones are not listed."""
    order = {"pending": 0, "skipped": 1, "stale": 1, "deferred": 2}
    rows = []
    for cid in pack.offered():
        st = state.status(cid)
        if st in order:
            rows.append((order[st], bool(pack.views[cid]["conflict_groups"]), pack.views[cid]["attribute"], cid))
    return [r[3] for r in sorted(rows)]


def pilot(pack) -> list:
    """The proposed pilot (D8): the offered, non-private candidates, as (id, key). Nothing private, instruction or
    template-default is ever in it."""
    return [(c, pack.views[c]["attribute"]) for c in pack.offered()]


def summary(pack, state) -> dict:
    return {"run": state.run, "records": len(state.records),
            "by_review_class": dict(sorted(Counter(v["review_class"] for v in pack.views.values()).items())),
            "offered": len(pack.offered()),
            "status": dict(sorted(Counter(state.status(c) for c in pack.offered()).items())),
            "conflict_choices": {g: r["choice"] for g, r in sorted(state.choices.items())}}


# ---- checks shared by lock and edit -------------------------------------------------------------

def _view(pack, cid):
    v = pack.views.get(cid)
    if v is None:
        raise ReviewError("unknown_candidate")
    return v


def _open_for_review(pack, state, cid):
    v = _view(pack, cid)
    if v["review_class"] != OFFER:
        raise ReviewError("not_offered", (v["review_class"],))
    if cid in state.final:
        raise ReviewError("already_reviewed", (state.final[cid]["action"],))
    return v


def _conflict_gate(pack, state, v, *, editing):
    """A candidate in a conflict group can be locked or edited only after the person has chosen, and only as the choice allows."""
    for gid in v["conflict_groups"]:
        g = pack.groups.get(gid)
        if g is None or g.get("scope") != "memory":
            continue
        ch = state.choices.get(gid)
        if ch is None:
            raise ReviewError("conflict_unresolved", (gid,))
        kind = ch["choice"]
        if kind == "neither":
            raise ReviewError("conflict_chose_neither", (gid,))
        if kind == "own" and not editing:
            raise ReviewError("conflict_choice_requires_edit", (gid,))
        if kind == "side" and ch.get("candidate") != v["id"]:
            raise ReviewError("conflict_not_chosen", (gid,))


def _key(state, v, key, external_taken):
    """The final key. A singleton key is fixed by the registry; a set key may be renamed but keeps its prefix (a goal stays a goal)."""
    proposed = K.canonical(v["attribute"])
    key = K.canonical(key if key is not None else v["attribute"])
    if K.kind_of(proposed) == K.SINGLETON and key != proposed:
        raise ReviewError("key_invalid", ("singleton_fixed",))
    if K.kind_of(proposed) == K.SET and key != proposed and \
            next((p for p in K.SET_PREFIXES if key.startswith(p)), None) != next(p for p in K.SET_PREFIXES if proposed.startswith(p)):
        raise ReviewError("key_invalid", ("prefix_change",))
    taken = {k: c for k, c in state.taken.items() if c != v["id"]}
    taken.update(external_taken or {})
    errors = K.validate_key(key, taken)
    if errors:
        raise ReviewError("key_invalid", errors)
    return key


def _clean_text(v, key, value, evidence):
    if not isinstance(value, str) or not isinstance(evidence, str):
        raise ReviewError("bad_value", ("not_text",))
    value, evidence = value.strip(), evidence.strip()
    if any(ord(c) < 32 or ord(c) == 127 for c in value + evidence):         # (newlines and tabs are control characters too)
        raise ReviewError("bad_value", ("control_characters",))
    if not 3 <= len(value) <= X.VALUE_MAX:
        raise ReviewError("bad_value", ("value_length",))
    if not G.evidence_usable(evidence) or len(evidence) < G.EVIDENCE_MIN:
        raise ReviewError("bad_value", ("evidence_unusable",))
    hit = X.screen_text(value + " " + evidence)
    if hit:
        raise ReviewError("screen_refused", (hit[0],))
    why = X.contact_or_third_party(key, value, evidence)
    if why:
        raise ReviewError("screen_refused", (why,))
    return value, evidence


def review_ref(run, cid, at) -> str:
    """The source reference a reviewed item would carry: `review:<run label>#<candidate>@<confirmation instant>`. The run is
    a short hash, never its digits, because a long run of digits looks like a card number to the gate's own secret screen."""
    return f"review:{X.sha(run)[:8]}#{cid}@{at}"


def build_item(run, cid, *, type, key, value, evidence, sensitivity, at, valid_from):
    """(the MemoryItem a review decision would store, its evidence). The one place that builds it: the gate dry-run in `lock`
    and `edit` and any later committer call this, so what was reviewed is what would be written. `at` is the confirmation
    instant (it is the item's created and updated time and part of its source reference); no supersession links are set here,
    because only the gate, against a real store, decides those."""
    item = M.MemoryItem(id=M.derive_id("user", key, value, at), type=type, subject="user", attribute=key, value=value,
                        created=at, updated=at, source=M.Source("user", review_ref(run, cid, at)),
                        confidence="stated", valid_from=valid_from, status="active", sensitivity=sensitivity)
    return item, evidence


def _dry_run(state, v, key, value, evidence, sensitivity, now, valid_from):
    item, evidence = build_item(state.run, v["id"], type=v["type"], key=key, value=value, evidence=evidence,
                                sensitivity=sensitivity, at=now, valid_from=valid_from)
    return G.evaluate(G.Proposal(item, evidence), (), now)


def _decide(pack, state, cid, *, action, now, problems, key, value, evidence, requested_sensitivity, external_taken):
    v = _open_for_review(pack, state, cid)
    if problems:
        raise ReviewError("stale_source", tuple(problems))
    _conflict_gate(pack, state, v, editing=(action == "edit"))
    key = _key(state, v, key, external_taken)
    value, evidence = _clean_text(v, key, value, evidence)
    computed = X.sensitivity_of(key, value, evidence)
    sensitivity = "private" if "private" in (computed, requested_sensitivity, v["sensitivity"]) else "normal"
    genuine = action == "lock" and v["origin"] == "transcript" and v["timestamp"]
    valid_from = v["timestamp"] if genuine else now
    d = _dry_run(state, v, key, value, evidence, sensitivity, now, valid_from)
    if d.decision == G.REQUIRE_CONFIRMATION:
        blocked = BLOCK_PRIVATE
    elif d.decision == G.ACCEPT:
        blocked = None
    else:
        raise ReviewError("gate_refused", (d.decision, d.reason))
    rec = {"action": action, "candidate": cid, "key": key, "key_proposed": v["attribute"], "type": v["type"],
           "registry_version": K.REGISTRY_VERSION, "sensitivity": sensitivity,
           "origin": {"kind": v["source_kind"], "ref": v["source_ref"], "where": v["where"], "line": v["line"],
                      "text_sha": v["text_sha"], "evidence_sha": X.sha(v["evidence"]) if v["evidence"] is not None else None,
                      "channel": v["channel"], "tier": v["evidence_tier"]},
           "adoption": {"by": "user", "channel": "tty", "at": now,
                        "kind": "adopted_note_text" if action == "lock" else "typed_by_user"},
           "value_fp": M.fingerprint(value), "value_len": len(value), "evidence_sha": X.sha(evidence),
           "valid_from": valid_from, "valid_from_basis": "transcript_turn" if genuine else "confirmation_instant",
           "gate": {"decision": d.decision, "reason": d.reason}, "commit_blocked": blocked, "source_check": []}
    private = None
    if sensitivity == "private":
        rec.update(value=None, evidence=None)
        private = {"value": value, "evidence": evidence}
    else:
        rec.update(value=value, evidence=evidence)
    return rec, private


# ---- the actions ---------------------------------------------------------------------------------

def lock(pack, state, cid, typed, *, now, problems=(), external_taken=None) -> Outcome:
    """Adopt the candidate exactly as shown. `typed` must be the word `lock`."""
    if typed != LOCK_WORD:
        raise ReviewError("lock_word")
    v = _view(pack, cid)
    if v["value"] is None or v["evidence"] is None:
        raise ReviewError("not_offered", (v["review_class"],))
    rec, private = _decide(pack, state, cid, action="lock", now=now, problems=problems, key=None, value=v["value"],
                           evidence=v["evidence"], requested_sensitivity=None, external_taken=external_taken)
    if rec["sensitivity"] != v["sensitivity"]:
        raise ReviewError("sensitivity_changed", (rec["sensitivity"],))
    return _finish(state, now, rec, private)


def edit(pack, state, cid, *, value, evidence=None, key=None, sensitivity=None, now, problems=(), external_taken=None) -> Outcome:
    """Record the person's own words as the value. The sentence defaults to the value. It can raise sensitivity, never lower it."""
    if sensitivity not in (None, "normal", "private"):
        raise ReviewError("bad_sensitivity")
    rec, private = _decide(pack, state, cid, action="edit", now=now, problems=problems, key=key, value=value,
                           evidence=value if evidence is None else evidence, requested_sensitivity=sensitivity,
                           external_taken=external_taken)
    return _finish(state, now, rec, private)


def _finish(state, now, rec, private) -> Outcome:
    out = new_record(state.records, now, rec)
    return Outcome(out, None if private is None else {"seq": out["seq"], **private})


def _simple(pack, state, cid, action, now, extra=None) -> Outcome:
    v = _open_for_review(pack, state, cid)
    fields = {"action": action, "candidate": cid, "key": v["attribute"]}
    fields.update(extra or {})
    return Outcome(new_record(state.records, now, fields))


def reject(pack, state, cid, reason, *, now) -> Outcome:
    if reason not in REJECT_REASONS:
        raise ReviewError("bad_reason")
    return _simple(pack, state, cid, "reject", now, {"reason": reason})


def skip(pack, state, cid, *, now) -> Outcome:
    return _simple(pack, state, cid, "skip", now)


def defer(pack, state, cid, *, now) -> Outcome:
    return _simple(pack, state, cid, "defer", now)


def stale(pack, state, cid, problems, *, now) -> Outcome:
    return _simple(pack, state, cid, "stale", now, {"source_check": sorted(problems)})


def conflict_choice(pack, state, gid, choice, *, now, label=None) -> Outcome:
    """Record how the person resolved a conflict group. Memory conflicts: side(label) / own / both / neither. Note-drift
    conflicts: only `drift` (acknowledged; the fix is theirs, in the ordinary notes). Nothing is ever defaulted."""
    g = pack.groups.get(gid)
    if g is None:
        raise ReviewError("unknown_conflict")
    if gid in state.choices:
        raise ReviewError("already_resolved", (gid,))
    if g["scope"] == "document_drift":
        if choice != "drift":
            raise ReviewError("bad_choice", ("drift_only",))
        return Outcome(new_record(state.records, now, {"action": "conflict_choice", "group": gid, "key": g["key"],
                                                       "choice": "drift", "candidate": None, "label": None}))
    if choice not in ("side", "own", "both", "neither"):
        raise ReviewError("bad_choice")
    cand = None
    if choice == "side":
        sides = {s["label"]: s for s in g["sides"]}
        if label not in sides:
            raise ReviewError("bad_choice", ("label",))
        wanted = [c for c in g["candidates"] if _matches_side(pack.views[c], sides[label], g)]
        cand = wanted[0] if wanted else None
        if cand is not None and pack.views[cand]["review_class"] != OFFER:
            raise ReviewError("side_not_reviewable", (pack.views[cand]["review_class"],))
        if cand is None:
            choice = "own"                  # the side the person chose has no candidate: they must type it themselves
    return Outcome(new_record(state.records, now, {"action": "conflict_choice", "group": gid, "key": g["key"],
                                                   "choice": choice, "candidate": cand, "label": label}))


def _matches_side(view, side, group) -> bool:
    """Does a side of a conflict point at this candidate's own source line?"""
    return side["path"] == view["where"] and side["line"] == view["line"] or \
        (group["kind"] == "generic" and side["label"] == view["id"])


def why(pack, state, cid) -> list:
    """Lines explaining a candidate: everything the person needs, nothing private."""
    v = _view(pack, cid)
    lines = [f"key        {v['key']}  ({v['registry']['kind']}, registry v{v['registry']['version']})",
             f"type       {v['type']}", f"class      {v['review_class']}   status {state.status(cid)}",
             f"tier       {v['evidence_tier']}   confidence {v['confidence']} (as found)",
             f"source     {v['source_kind']}  {v['source_ref']}", f"hash       {v['text_sha'][:16]}",
             f"gate       as found: {v['gate']['decision']} {v['gate']['reason']}; if you adopt it: "
             f"{v['if_affirmed']['decision']} {v['if_affirmed']['reason']}"]
    if v["source_date"]:
        lines.append(f"date seen  {v['source_date']} (shown as evidence only)")
    if v["redacted"]:
        lines.append("value      [PRIVATE - not shown here]")
    else:
        lines += [f"value      {v['value']}", f"evidence   {v['evidence']}"]
    for gid in v["conflict_groups"]:
        g = pack.groups.get(gid, {})
        ch = state.choices.get(gid)
        lines.append(f"conflict   {gid} {g.get('key')} [{g.get('scope')}] " + (f"resolved: {ch['choice']}" if ch else "UNRESOLVED"))
    return lines


# ---- replay: does the ledger say only what this machine could have written? -----------------------------------------

def replay_validate(pack, records, pack_sha, private=None) -> list:
    """Problem codes for a ledger (empty = it replays cleanly against this pack). `verify_chain` shows only that no record
    was altered, reordered or dropped; it cannot show that a record was ever LEGITIMATE, because anyone can build a record
    whose hash chains. This replays the ledger from its start: record i is regenerated by the same function that writes it
    (`lock`, `edit`, `reject`, `skip`, `defer`, `stale`, `conflict_choice`, `session`) against the state of records [0, i),
    with the record's own `at` as the clock, and must be IDENTICAL. So a record for a candidate the pack never offered, an
    action after a final decision, a lock before a conflict was chosen, a value the gate refuses, a key outside the registry,
    a changed value or an extra field is caught. `private` maps a record's seq to {"value", "evidence"} (the 0600 side
    file); a private edit cannot be replayed without it.

    WHAT THIS CANNOT PROVE: that a person made the decision. A forged record that is consistent with a valid pack, built by
    calling `lock` or `edit` directly, replays cleanly, because code running as the same user can do exactly what the review
    CLI does. It is a consistency check, not an authentication."""
    if not isinstance(records, (list, tuple)) or not all(isinstance(r, dict) for r in records):
        return ["malformed"]
    problems = verify_chain(records)
    if problems:
        return problems
    if not records:
        return ["empty"]
    head = records[0]
    try:
        want = open_record([], head["at"], run=head["run"], pack_sha=pack_sha, pack_format=X.PACK_FORMAT)
    except (KeyError, TypeError, ValueError, ReviewError):
        return ["replay:0:malformed"]
    if head != want:
        return ["replay:0:mismatch"]
    private = private or {}
    for i in range(1, len(records)):
        rec = records[i]
        try:
            state = State(records[:i])
            again = _regenerate(pack, state, rec, head, pack_sha, private.get(i))
        except ReviewError as e:
            problems.append(f"replay:{i}:{e.code}")
            continue
        except (KeyError, TypeError, ValueError, AttributeError):
            problems.append(f"replay:{i}:malformed")
            continue
        if again != rec:
            problems.append(f"replay:{i}:mismatch")
    return problems


def _regenerate(pack, state, rec, head, pack_sha, private):
    """The record the machine would write for `rec`'s own inputs at `rec`'s own instant."""
    action, at = rec.get("action"), rec["at"]
    if action == "session":
        return session_record(state.records, at, run=head["run"], pack_sha=pack_sha)
    if action == "conflict_choice":
        return conflict_choice(pack, state, rec["group"], rec["choice"], now=at, label=rec.get("label")).record
    if action not in ("lock", "edit", "reject", "skip", "defer", "stale"):
        raise ReviewError("unknown_action")
    cid = rec["candidate"]
    if action == "lock":
        return lock(pack, state, cid, LOCK_WORD, now=at).record
    if action == "edit":
        value, evidence = rec["value"], rec["evidence"]
        if rec["sensitivity"] == "private" and value is None:
            if not private:
                raise ReviewError("private_missing")
            value, evidence = private["value"], private["evidence"]
        return edit(pack, state, cid, value=value, evidence=evidence, key=rec["key"],
                    sensitivity="private" if rec["sensitivity"] == "private" else None, now=at).record
    if action == "reject":
        return reject(pack, state, cid, rec["reason"], now=at).record
    if action in ("skip", "defer"):
        return (skip if action == "skip" else defer)(pack, state, cid, now=at).record
    return stale(pack, state, cid, rec["source_check"], now=at).record
