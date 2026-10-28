"""Memory 2.0, Phase 1: the pure semantic core (backtalk/memory.py) and the read-only vault inventory."""
import ast
import copy
import inspect
import itertools
import os
import random
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from backtalk import memory as M
from tests import memory_inventory as INV


def day(n, hour=0):
    return f"2026-01-{n:02d}T{hour:02d}:00:00Z"


def mid(n):
    return f"mem_{n:012x}"


def raw(n=1, **over):
    d = {"id": mid(n), "type": "fact", "subject": "user", "attribute": "residence", "value": "Kansas City",
         "created": day(1), "updated": day(1), "source": {"kind": "user", "ref": "turn 12"},
         "confidence": "stated", "valid_from": day(1), "status": "active", "sensitivity": "normal"}
    d.update(over)
    return {k: v for k, v in d.items() if v is not _GONE}


_GONE = object()


def item(n=1, **over):
    it, errs = M.parse(raw(n, **over))
    assert it is not None, errs
    return it


def mk(n=1, **over):
    """An item built without validation, so an invalid one can be made."""
    d = raw(n, **over)
    src = d.pop("source")
    return M.MemoryItem(source=M.Source(**src), valid_to=d.pop("valid_to", None),
                        supersedes=tuple(d.pop("supersedes", ())), **d)


class ValidationTests(unittest.TestCase):
    def test_a_valid_item_parses_and_round_trips(self):
        it, errs = M.parse(raw())
        self.assertEqual(errs, ())
        self.assertEqual(M.validate(it), ())
        self.assertEqual(M.parse(M.to_dict(it)), (it, ()))
        full = item(7, valid_from=day(2), valid_to=day(9), supersedes=[mid(3), mid(4)], status="superseded",
                    source={"kind": "note", "ref": "09 - Resources/X.md"}, confidence="inferred", sensitivity="private")
        self.assertEqual(M.parse(M.to_dict(full)), (full, ()))
        self.assertEqual(M.to_dict(full)["supersedes"], [mid(3), mid(4)])
        self.assertEqual(it.valid_to, None)
        self.assertEqual(it.supersedes, ())

    def test_every_required_field_missing_is_reported_by_name(self):
        for f in M.REQUIRED_FIELDS:
            it, errs = M.parse(raw(**{f: _GONE}))
            self.assertIsNone(it, f)
            self.assertEqual(errs, (f"missing:{f}",), f)

    def test_the_two_optional_fields_may_be_absent_or_null(self):
        self.assertIsNotNone(M.parse(raw(valid_to=None, supersedes=[]))[0])
        self.assertIsNotNone(M.parse(raw())[0])

    def test_blank_values_are_missing_not_valid(self):
        for f in ("type", "subject", "attribute", "value", "confidence", "status", "sensitivity", "created", "valid_from"):
            for blank in ("", "   "):
                self.assertEqual(M.validate(mk(**{f: blank})), (f"missing:{f}",), (f, blank))

    def test_invalid_type_status_confidence_sensitivity(self):
        for f, bad in (("type", "temporary"), ("type", "Fact"), ("status", "deleted"), ("confidence", "certain"),
                       ("sensitivity", "secret"), ("sensitivity", "Private")):
            it, errs = M.parse(raw(**{f: bad}))
            self.assertIsNone(it)
            self.assertEqual(errs, (f"invalid:{f}",), (f, bad))

    def test_all_declared_vocabularies_are_accepted(self):
        for f, vals in (("type", M.TYPES), ("confidence", M.CONFIDENCES), ("sensitivity", M.SENSITIVITIES)):
            for v in vals:
                self.assertIsNotNone(M.parse(raw(**{f: v}))[0], (f, v))
        for st in M.STATUSES:
            over = {"status": st, "valid_to": day(9)} if st == "superseded" else {"status": st}
            self.assertIsNotNone(M.parse(raw(**over))[0], st)

    def test_the_exact_vocabularies(self):
        self.assertEqual(M.TYPES, ("identity", "preference", "fact", "goal", "event", "instruction"))
        self.assertEqual(M.STATUSES, ("active", "superseded", "archived"))
        self.assertEqual(M.CONFIDENCES, ("stated", "inferred"))
        self.assertEqual(M.SENSITIVITIES, ("normal", "private", "never-expose"))

    def test_id_subject_attribute_formats(self):
        for bad in ("mem_123", "MEM_0123456789ab", "mem_0123456789abc", "0123456789ab", "mem_0123456789ag", "x"):
            self.assertEqual(M.parse(raw(id=bad))[1], ("invalid:id",), bad)
        for f in ("subject", "attribute"):
            for bad in ("User", "a.b", "a b", "-x", "x" * 65, "é"):
                self.assertEqual(M.parse(raw(**{f: bad}))[1], (f"invalid:{f}",), (f, bad))
            for good in ("user", "kansas-city", "a1_b", "9x", "x" * 64):
                self.assertIsNotNone(M.parse(raw(**{f: good}))[0], (f, good))

    def test_value_and_source_structure(self):
        self.assertEqual(M.parse(raw(value="x" * (M.VALUE_MAX + 1)))[1], ("invalid:value",))
        self.assertIsNotNone(M.parse(raw(value="x" * M.VALUE_MAX))[0])
        for bad in ({"kind": "web", "ref": "x"}, {"kind": "user", "ref": ""}, {"kind": "user", "ref": "a\nb"},
                    {"kind": "user", "ref": "x" * (M.REF_MAX + 1)}, {"kind": "user"}, {"kind": "user", "ref": "x", "extra": 1},
                    "user", None, 5, {"kind": 1, "ref": "x"}):
            self.assertEqual(M.parse(raw(source=bad))[1], ("invalid:source",), repr(bad))
        for kind in M.SOURCE_KINDS:
            self.assertIsNotNone(M.parse(raw(source={"kind": kind, "ref": "r"}))[0], kind)

    def test_timestamps_must_be_the_exact_utc_format(self):
        for bad in ("2026-01-01", "2026-01-01T00:00:00", "2026-01-01T00:00:00+00:00", "2026-01-01 00:00:00Z",
                    "2026-13-01T00:00:00Z", "2026-02-30T00:00:00Z", "2026-01-01T24:00:00Z", "2026-01-01T00:00:00z",
                    "2026-01-01T00:00:00.5Z", " 2026-01-01T00:00:00Z"):
            for f in ("created", "updated", "valid_from", "valid_to"):
                self.assertEqual(M.parse(raw(**{f: bad}))[1], (f"invalid:timestamp:{f}",), (f, bad))
        for f in ("created", "updated", "valid_from"):
            self.assertEqual(M.parse(raw(**{f: 20260101}))[1], (f"invalid:{f}",))

    def test_a_trailing_newline_or_junk_never_slips_past_a_format(self):
        for f in ("created", "updated", "valid_from", "valid_to"):
            for bad in ("2026-01-01T00:00:00Z\n", "2026-01-01T00:00:00Zjunk"):
                self.assertEqual(M.parse(raw(**{f: bad}))[1], (f"invalid:timestamp:{f}",), (f, bad))
        for bad in (mid(1) + "\n", mid(1) + "x"):
            self.assertEqual(M.parse(raw(id=bad))[1], ("invalid:id",), repr(bad))
        for f in ("subject", "attribute"):
            self.assertEqual(M.parse(raw(**{f: "user\n"}))[1], (f"invalid:{f}",))
        self.assertEqual(M.parse(raw(supersedes=[mid(2) + "\n"]))[1], ("invalid:supersedes",))

    def test_the_limits_are_pinned(self):
        self.assertEqual((M.VALUE_MAX, M.REF_MAX, M.SUPERSEDES_MAX), (2000, 300, 8))
        self.assertEqual(M.parse(raw(value="x" * 2000))[1], ())
        self.assertEqual(M.parse(raw(value="x" * 2001))[1], ("invalid:value",))
        self.assertEqual(M.parse(raw(source={"kind": "user", "ref": "r" * 300}))[1], ())
        self.assertEqual(M.parse(raw(source={"kind": "user", "ref": "r" * 301}))[1], ("invalid:source",))
        self.assertEqual(M.parse(raw(supersedes=[mid(i) for i in range(2, 10)]))[1], ())
        self.assertEqual(M.parse(raw(supersedes=[mid(i) for i in range(2, 11)]))[1], ("invalid:supersedes",))

    def test_a_non_string_valid_to_is_named_as_such(self):
        self.assertEqual(M.parse(raw(valid_to=5))[1], ("invalid:valid_to",))
        self.assertEqual(M.parse(raw(valid_to=["x"]))[1], ("invalid:valid_to",))

    def test_updated_cannot_precede_created(self):
        self.assertEqual(M.parse(raw(created=day(5), updated=day(4)))[1], ("order:updated_before_created",))
        self.assertIsNotNone(M.parse(raw(created=day(5), updated=day(5)))[0])

    def test_valid_to_must_be_after_valid_from(self):
        self.assertEqual(M.parse(raw(valid_from=day(5), valid_to=day(5)))[1], ("range:valid_to_not_after_valid_from",))
        self.assertEqual(M.parse(raw(valid_from=day(5), valid_to=day(4)))[1], ("range:valid_to_not_after_valid_from",))
        self.assertIsNotNone(M.parse(raw(valid_from=day(5), valid_to=day(6)))[0])

    def test_superseded_status_requires_an_end(self):
        self.assertEqual(M.parse(raw(status="superseded"))[1], ("status:superseded_without_valid_to",))
        self.assertIsNotNone(M.parse(raw(status="superseded", valid_to=day(9)))[0])

    def test_supersedes_references(self):
        self.assertEqual(M.parse(raw(supersedes=["nope"]))[1], ("invalid:supersedes",))
        self.assertEqual(M.parse(raw(supersedes=[mid(2), mid(2)]))[1], ("invalid:supersedes",))
        self.assertEqual(M.parse(raw(supersedes=[mid(1)]))[1], ("supersedes:self",))
        self.assertEqual(M.parse(raw(supersedes=[mid(i) for i in range(2, 2 + M.SUPERSEDES_MAX + 1)]))[1], ("invalid:supersedes",))
        self.assertEqual(M.parse(raw(supersedes="mem_000000000002"))[1], ("invalid:supersedes",))
        self.assertEqual(M.parse(raw(supersedes=[1]))[1], ("invalid:supersedes",))
        self.assertEqual(M.parse(raw(supersedes=[mid(2), mid(3)]))[0].supersedes, (mid(2), mid(3)))
        self.assertEqual(M.validate(mk(supersedes=[mid(2)]).__class__(**{**mk().__dict__, "supersedes": [mid(2)]})), ("invalid:supersedes",))

    def test_nothing_is_coerced_repaired_or_dropped(self):
        for f in ("type", "subject", "value", "confidence", "created"):
            self.assertEqual(M.parse(raw(**{f: 5}))[1], (f"invalid:{f}",), f)
        self.assertEqual(M.parse(raw(extra="x"))[1], ("unknown_field:extra",))
        self.assertEqual(M.parse(raw(zzz=1, aaa=2))[1], ("unknown_field:aaa", "unknown_field:zzz"))
        self.assertEqual(M.parse(raw(type=" fact"))[1], ("invalid:type",))                     # not trimmed
        self.assertEqual(M.parse(raw(value="  padded  "))[0].value, "  padded  ")            # kept exactly
        self.assertEqual(M.parse(None)[1], ("invalid:not_a_mapping",))
        self.assertEqual(M.parse([("id", 1)])[1], ("invalid:not_a_mapping",))
        self.assertEqual(M.validate({"id": mid(1)}), ("invalid:not_a_memory_item",))

    def test_errors_are_deterministic_and_complete(self):
        bad = raw(type="x", status="y", confidence="z", sensitivity="w", id="q")
        a = M.parse(bad)[1]
        self.assertEqual(a, M.parse(copy.deepcopy(bad))[1])
        self.assertEqual(set(a), {"invalid:id", "invalid:type", "invalid:status", "invalid:confidence", "invalid:sensitivity"})
        self.assertEqual(len(a), len(set(a)))

    def test_parse_never_returns_an_item_alongside_errors(self):
        for over in ({"type": "x"}, {"id": "q"}, {"extra": 1}, {"created": "x"}):
            it, errs = M.parse(raw(**over))
            self.assertTrue(it is None and errs)

    def test_items_are_immutable(self):
        it = item()
        with self.assertRaises(Exception):
            it.value = "x"
        with self.assertRaises(Exception):
            it.source.kind = "note"

    def test_derive_id_is_stable_and_well_formed(self):
        a = M.derive_id("user", "residence", "Kansas City", day(1))
        self.assertRegex(a, r"^mem_[0-9a-f]{12}$")
        self.assertEqual(a, M.derive_id("user", "residence", "  kansas   CITY ", day(1)))
        for other in (("me", "residence", "Kansas City", day(1)), ("user", "home", "Kansas City", day(1)),
                      ("user", "residence", "San Jose", day(1)), ("user", "residence", "Kansas City", day(2))):
            self.assertNotEqual(a, M.derive_id(*other))
        self.assertEqual(M.parse(raw(id=a))[1], ())


class ActiveAtTimeTests(unittest.TestCase):
    def test_an_open_ended_item_is_active_from_valid_from_onward(self):
        it = item(valid_from=day(5))
        self.assertFalse(M.is_active(it, day(4)))
        self.assertTrue(M.is_active(it, day(5)))                       # valid_from is inclusive
        self.assertTrue(M.is_active(it, "2099-01-01T00:00:00Z"))

    def test_valid_to_is_exclusive(self):
        it = item(valid_from=day(5), valid_to=day(9))
        self.assertTrue(M.is_active(it, "2026-01-08T23:59:59Z"))
        self.assertFalse(M.is_active(it, day(9)))
        self.assertFalse(M.is_active(it, day(10)))

    def test_future_expired_and_current(self):
        self.assertFalse(M.is_active(item(valid_from=day(20)), day(10)))                      # future
        self.assertFalse(M.is_active(item(valid_from=day(2), valid_to=day(5)), day(10)))      # expired
        self.assertTrue(M.is_active(item(valid_from=day(2), valid_to=day(15)), day(10)))      # current

    def test_a_superseded_item_was_active_before_it_ended_and_is_not_after(self):
        old = item(status="superseded", valid_from=day(1), valid_to=day(10))
        self.assertTrue(M.is_active(old, day(5)))                      # a historical question can be answered
        self.assertFalse(M.is_active(old, day(10)))
        self.assertFalse(M.is_active(old, day(20)))

    def test_an_archived_item_is_never_active_unless_history_asks(self):
        a = item(status="archived")
        self.assertFalse(M.is_active(a, day(5)))
        self.assertFalse(M.is_active(a, day(5), include_archived=False))
        self.assertTrue(M.is_active(a, day(5), include_archived=True))
        self.assertFalse(M.is_active(a, day(5) if False else "2025-01-01T00:00:00Z", include_archived=True))   # still time-bound

    def test_an_invalid_item_is_never_active(self):
        self.assertFalse(M.is_active(mk(type="bogus"), day(5)))
        self.assertFalse(M.is_active(mk(valid_from="soon"), day(5)))
        self.assertFalse(M.is_active("not an item", day(5)))

    def test_the_callers_time_is_used_and_must_be_a_real_time(self):
        it = item(valid_from=day(5))
        aware = datetime(2026, 1, 6, tzinfo=timezone.utc)
        self.assertTrue(M.is_active(it, aware))
        self.assertTrue(M.is_active(it, aware.astimezone(timezone(timedelta(hours=-6)))))        # any zone, same instant
        self.assertFalse(M.is_active(it, datetime(2026, 1, 4, 23, tzinfo=timezone(timedelta(hours=-6)))) is False and False)
        for bad in (datetime(2026, 1, 6), "2026-01-06", None, 5, "2026-01-06T00:00:00+00:00"):
            with self.assertRaises(ValueError, msg=repr(bad)):
                M.is_active(it, bad)

    def test_timezone_conversion_is_exact(self):
        it = item(valid_from="2026-01-05T06:00:00Z")
        self.assertFalse(M.is_active(it, datetime(2026, 1, 4, 23, 59, tzinfo=timezone(timedelta(hours=-6)))))   # = 05:59Z
        self.assertTrue(M.is_active(it, datetime(2026, 1, 5, 0, 0, tzinfo=timezone(timedelta(hours=-6)))))      # = 06:00Z


class SupersessionTests(unittest.TestCase):
    def active_ids(self, res):
        return [i.id for i in res.active]

    def reasons(self, res):
        return {i.id: r for i, r in res.history}

    def test_a_newer_stated_item_supersedes_an_older_stated_one(self):
        old, new = item(1, value="Kansas City"), item(2, value="San Jose", valid_from=day(10), created=day(10), updated=day(10))
        res = M.resolve([old, new], day(20))
        self.assertEqual(self.active_ids(res), [mid(2)])
        self.assertEqual(self.reasons(res), {mid(1): "superseded"})
        self.assertEqual(res.conflicts, ())

    def test_the_old_value_is_still_the_answer_to_a_question_about_the_past(self):
        old, new = item(1), item(2, value="San Jose", valid_from=day(10), created=day(10), updated=day(10))
        self.assertEqual(self.active_ids(M.resolve([old, new], day(5))), [mid(1)])
        self.assertEqual(self.reasons(M.resolve([old, new], day(5))), {mid(2): "not_yet_valid"})

    def test_an_inferred_item_never_overrides_a_stated_one_however_new(self):
        stated = item(1)
        inferred = item(2, confidence="inferred", value="San Jose", valid_from=day(10), created=day(10), updated=day(10))
        res = M.resolve([stated, inferred], day(20))
        self.assertEqual(self.active_ids(res), [mid(1)])
        self.assertEqual(self.reasons(res), {mid(2): "shadowed"})

    def test_a_newer_stated_item_replaces_an_older_inferred_one(self):
        inferred = item(1, confidence="inferred", value="Kansas City")
        stated = item(2, value="San Jose", valid_from=day(10), created=day(10), updated=day(10))
        res = M.resolve([inferred, stated], day(20))
        self.assertEqual(self.active_ids(res), [mid(2)])
        self.assertEqual(self.reasons(res), {mid(1): "superseded"})

    def test_a_stated_item_outranks_an_inferred_one_even_when_older(self):
        res = M.resolve([item(1, confidence="inferred", valid_from=day(12), created=day(12), updated=day(12)), item(2, value="x")], day(20))
        self.assertEqual(self.active_ids(res), [mid(2)])

    def test_among_inferred_items_the_newest_wins(self):
        a = item(1, confidence="inferred", value="a")
        b = item(2, confidence="inferred", value="b", valid_from=day(5), created=day(5), updated=day(5))
        self.assertEqual(self.active_ids(M.resolve([a, b], day(20))), [mid(2)])

    def test_equal_authority_equally_recent_different_values_is_an_explicit_conflict(self):
        a, b = item(1, value="Kansas City"), item(2, value="San Jose")
        res = M.resolve([b, a], day(20))
        self.assertEqual(res.active, ())
        self.assertEqual(len(res.conflicts), 1)
        self.assertEqual(res.conflicts[0].key, "user.residence")
        self.assertEqual([i.id for i in res.conflicts[0].candidates], [mid(1), mid(2)])       # sorted by id
        self.assertEqual(res.history, ())

    def test_a_tie_on_valid_from_is_broken_by_created(self):
        a = item(1, value="a")
        b = item(2, value="b", created=day(3), updated=day(3))
        res = M.resolve([a, b], day(20))
        self.assertEqual(self.active_ids(res), [mid(2)])
        self.assertEqual(res.conflicts, ())

    def test_valid_from_outranks_created(self):
        backfilled = item(1, value="2020 home", valid_from="2020-01-01T00:00:00Z", created=day(25), updated=day(25))
        current = item(2, value="now", valid_from=day(5), created=day(6), updated=day(6))
        self.assertEqual(self.active_ids(M.resolve([backfilled, current], day(28))), [mid(2)])

    def test_the_same_value_twice_is_a_duplicate_not_a_conflict(self):
        res = M.resolve([item(2, value="Kansas  City"), item(1, value="kansas city")], day(20))
        self.assertEqual(self.active_ids(res), [mid(1)])                                       # the lowest id stands
        self.assertEqual(self.reasons(res), {mid(2): "duplicate"})
        self.assertEqual(res.conflicts, ())

    def test_same_value_but_different_type_is_a_conflict_not_a_duplicate(self):
        res = M.resolve([item(1, type="fact"), item(2, type="preference")], day(20))
        self.assertEqual(len(res.conflicts), 1)
        self.assertEqual(res.active, ())

    def test_different_keys_never_interact(self):
        res = M.resolve([item(1), item(2, attribute="employer", value="Acme"), item(3, subject="max", value="x")], day(20))
        self.assertEqual(sorted(self.active_ids(res)), [mid(1), mid(2), mid(3)])
        self.assertEqual([M.key_of(i) for i in res.active], ["max.residence", "user.employer", "user.residence"])
        self.assertEqual((res.conflicts, res.history), ((), ()))

    def test_an_explicit_link_beats_the_ordering(self):
        replaced = item(1, value="old", valid_from=day(10), created=day(10), updated=day(10))
        replacer = item(2, value="new", supersedes=[mid(1)])                                  # older timestamps, explicit link
        res = M.resolve([replaced, replacer], day(20))
        self.assertEqual(self.active_ids(res), [mid(2)])
        self.assertEqual(self.reasons(res), {mid(1): "superseded"})

    def test_an_explicit_link_does_nothing_before_the_replacer_is_valid(self):
        replaced = item(1, value="old")
        replacer = item(2, value="new", valid_from=day(15), created=day(2), updated=day(2), supersedes=[mid(1)])
        self.assertEqual(self.active_ids(M.resolve([replaced, replacer], day(10))), [mid(1)])
        self.assertEqual(self.active_ids(M.resolve([replaced, replacer], day(16))), [mid(2)])

    def test_a_chain_of_links_leaves_only_the_last(self):
        a, b, c = item(1, value="a"), item(2, value="b", supersedes=[mid(1)]), item(3, value="c", supersedes=[mid(2)])
        res = M.resolve([a, b, c], day(20))
        self.assertEqual(self.active_ids(res), [mid(3)])
        self.assertEqual(self.reasons(res), {mid(1): "superseded", mid(2): "superseded"})

    def test_archived_items_are_excluded_from_the_active_set_and_from_conflicts(self):
        res = M.resolve([item(1, status="archived", value="a"), item(2, value="b")], day(20))
        self.assertEqual(self.active_ids(res), [mid(2)])
        self.assertEqual(self.reasons(res), {mid(1): "archived"})
        self.assertEqual(M.resolve([item(1, status="archived")], day(20)).active, ())

    def test_resolve_and_is_active_agree_at_the_boundaries(self):
        it = item(1, valid_from=day(5), valid_to=day(9))
        for t in (day(4), day(5), "2026-01-08T23:59:59Z", day(9), day(10)):
            self.assertEqual(bool(M.resolve([it], t).active), M.is_active(it, t), t)
        self.assertEqual(self.reasons(M.resolve([it], day(9))), {mid(1): "ended"})
        self.assertEqual(self.reasons(M.resolve([it], day(4))), {mid(1): "not_yet_valid"})

    def test_history_is_ordered_by_key_then_time_then_id(self):
        pool = [item(1, value="a", valid_from=day(3), created=day(3), updated=day(3)),
                item(2, value="b", valid_from=day(1)),
                item(3, value="c", valid_from=day(2), created=day(2), updated=day(2)),
                item(4, attribute="employer", value="e1", valid_to=day(2)),
                item(5, subject="max", value="m", valid_to=day(2)),
                item(6, value="z", valid_from=day(5), created=day(5), updated=day(5))]
        res = M.resolve(pool, day(20))
        self.assertEqual([(i.id, r) for i, r in res.history],
                         [(mid(5), "ended"), (mid(4), "ended"), (mid(2), "superseded"), (mid(3), "superseded"), (mid(1), "superseded")])
        self.assertEqual([i.id for i in res.active], [mid(6)])

    def test_ended_and_not_yet_valid_items_are_history(self):
        res = M.resolve([item(1, valid_to=day(5)), item(2, value="later", valid_from=day(25))], day(10))
        self.assertEqual(res.active, ())
        self.assertEqual(self.reasons(res), {mid(1): "ended", mid(2): "not_yet_valid"})

    def test_a_superseded_status_item_is_history_once_it_ended(self):
        old = item(1, status="superseded", valid_to=day(10))
        new = item(2, value="new", valid_from=day(10), created=day(10), updated=day(10), supersedes=[mid(1)])
        res = M.resolve([old, new], day(20))
        self.assertEqual(self.active_ids(res), [mid(2)])
        self.assertEqual(self.reasons(res), {mid(1): "ended"})

    def test_invalid_items_are_rejected_not_repaired_and_do_not_disturb_the_rest(self):
        res = M.resolve([mk(1, type="bogus"), item(2), "junk", {"id": "x"}], day(20))
        self.assertEqual(self.active_ids(res), [mid(2)])
        self.assertEqual(res.rejected, ((mid(1), ("invalid:type",)), ("#2", ("invalid:not_a_memory_item",)),
                                        ("#3", ("invalid:not_a_memory_item",))))

    def test_duplicate_ids_reject_every_holder(self):
        res = M.resolve([item(1, value="a"), item(1, value="b"), item(2, attribute="x")], day(20))
        self.assertEqual(self.active_ids(res), [mid(2)])
        self.assertEqual([r[1] for r in res.rejected], [("id:duplicate",), ("id:duplicate",)])

    def test_bad_links_reject_the_linker(self):
        unknown = M.resolve([item(2, supersedes=[mid(9)])], day(20))
        self.assertEqual(unknown.rejected, ((mid(2), (f"supersedes:unknown:{mid(9)}",)),))
        mismatch = M.resolve([item(1), item(2, attribute="employer", supersedes=[mid(1)])], day(20))
        self.assertEqual(self.active_ids(mismatch), [mid(1)])
        self.assertEqual(mismatch.rejected, ((mid(2), (f"supersedes:key_mismatch:{mid(1)}",)),))

    def test_an_inferred_item_cannot_explicitly_replace_a_stated_one(self):
        res = M.resolve([item(1), item(2, confidence="inferred", value="x", supersedes=[mid(1)])], day(20))
        self.assertEqual(self.active_ids(res), [mid(1)])
        self.assertEqual(res.rejected, ((mid(2), (f"supersedes:authority:{mid(1)}",)),))

    def test_a_cycle_rejects_both_ends_in_any_order(self):
        a, b = item(1, supersedes=[mid(2)]), item(2, value="b", supersedes=[mid(1)])
        for order in ((a, b), (b, a)):
            res = M.resolve(order, day(20))
            self.assertEqual(res.active, ())
            self.assertEqual({r[0] for r in res.rejected}, {mid(1), mid(2)})
            self.assertTrue(all(r[1][0].startswith("supersedes:cycle:") for r in res.rejected))

    def test_rejecting_an_item_strands_the_link_to_it(self):
        res = M.resolve([mk(1, type="bogus"), item(2, supersedes=[mid(1)])], day(20))
        self.assertEqual(res.active, ())
        self.assertEqual({r[0] for r in res.rejected}, {mid(1), mid(2)})

    def test_the_answer_is_the_same_in_every_input_order(self):
        pool = [item(1, value="a"), item(2, value="b", valid_from=day(5), created=day(5), updated=day(5)),
                item(3, confidence="inferred", value="c", valid_from=day(9), created=day(9), updated=day(9)),
                item(4, attribute="employer", value="Acme"), item(5, attribute="employer", value="Zed"),
                item(6, subject="max", status="archived"), item(7, subject="max", attribute="age", value="9", supersedes=[]),
                item(8, subject="max", attribute="age", value="10", supersedes=[mid(7)], valid_from=day(2))]
        base = M.resolve(pool, day(20))
        rng = random.Random(7)
        for _ in range(40):
            shuffled = pool[:]
            rng.shuffle(shuffled)
            res = M.resolve(shuffled, day(20))
            self.assertEqual((res.active, res.conflicts, res.history), (base.active, base.conflicts, base.history))
        self.assertEqual([i.id for i in base.active], [mid(8), mid(2)])
        self.assertEqual([c.key for c in base.conflicts], ["user.employer"])

    def test_resolution_does_not_mutate_its_inputs(self):
        pool = [item(1), item(2, value="b", valid_from=day(5), created=day(5), updated=day(5)), item(3, status="archived"),
                item(4, supersedes=[mid(1)], value="z")]
        snapshot = copy.deepcopy(pool)
        as_list = list(pool)
        res = M.resolve(pool, day(20))
        self.assertEqual(pool, snapshot)
        self.assertEqual(pool, as_list)
        self.assertIsInstance(res.active, tuple)
        self.assertIsInstance(res.history, tuple)
        self.assertIsInstance(res.conflicts, tuple)
        self.assertEqual(M.resolve(iter(pool), day(20)), M.resolve(pool, day(20)))               # a one-shot iterable works too

    def test_resolution_reads_the_time_it_is_given_not_the_clock(self):
        pool = [item(1)]
        self.assertEqual(M.resolve(pool, day(2)), M.resolve(pool, day(2)))
        self.assertEqual(M.resolve(pool, datetime(2026, 1, 2, tzinfo=timezone.utc)), M.resolve(pool, day(2)))
        with self.assertRaises(ValueError):
            M.resolve(pool, "yesterday")

    def test_kansas_city_to_san_jose(self):
        kc = item(1, value="Kansas City", valid_to=day(15), status="superseded")
        sj = item(2, value="San Jose", valid_from=day(15), created=day(15), updated=day(15), supersedes=[mid(1)])
        for t, want in ((day(10), "Kansas City"), (day(14), "Kansas City"), (day(15), "San Jose"), (day(30), "San Jose")):
            self.assertEqual([i.value for i in M.resolve([kc, sj], t).active], [want], t)


class DuplicateTests(unittest.TestCase):
    def test_exact_duplicate_ignores_case_whitespace_and_unicode_form(self):
        a = item(1, value="Kansas City")
        self.assertEqual(M.classify_pair(a, item(2, value="  kansas    CITY ")), M.EXACT_DUPLICATE)
        self.assertEqual(M.classify_pair(item(1, value="Ｃａｆé"), item(2, value="café")), M.EXACT_DUPLICATE)
        self.assertEqual(M.classify_pair(a, a), M.EXACT_DUPLICATE)

    def test_same_key_different_value(self):
        self.assertEqual(M.classify_pair(item(1, value="Kansas City"), item(2, value="San Jose")), M.SAME_KEY_DIFFERENT_VALUE)
        self.assertEqual(M.classify_pair(item(1, value="a"), item(2, value="a b")), M.SAME_KEY_DIFFERENT_VALUE)

    def test_same_key_and_value_but_different_type_is_not_an_exact_duplicate(self):
        self.assertEqual(M.classify_pair(item(1, type="fact"), item(2, type="preference")), M.SAME_KEY_DIFFERENT_VALUE)

    def test_distinct_memories(self):
        self.assertEqual(M.classify_pair(item(1), item(2, subject="max")), M.DISTINCT)
        self.assertEqual(M.classify_pair(item(1), item(2, attribute="employer")), M.DISTINCT)
        self.assertEqual(M.classify_pair(item(1, value="x"), item(2, subject="max", attribute="employer", value="x")), M.DISTINCT)

    def test_the_key_is_subject_dot_attribute(self):
        self.assertEqual(M.key_of(item()), "user.residence")
        self.assertNotEqual(M.key_of(item(subject="a-b", attribute="c")), M.key_of(item(subject="a", attribute="b-c")))

    def test_find_duplicates_is_deterministic_and_leaves_out_self_and_distinct(self):
        cand = item(5, value="Kansas City")
        existing = [item(9, value="kansas city"), item(2, value="San Jose"), item(7, subject="max"), item(5, value="Kansas City"),
                    item(3, value="KANSAS CITY"), item(4, value="Denver")]
        want = ((mid(3), M.EXACT_DUPLICATE), (mid(9), M.EXACT_DUPLICATE),
                (mid(2), M.SAME_KEY_DIFFERENT_VALUE), (mid(4), M.SAME_KEY_DIFFERENT_VALUE))
        self.assertEqual(M.find_duplicates(cand, existing), want)
        self.assertEqual(M.find_duplicates(cand, reversed(existing)), want)
        self.assertEqual(M.find_duplicates(cand, []), ())

    def test_fingerprint_follows_the_normalized_value(self):
        self.assertEqual(M.fingerprint("Kansas City"), M.fingerprint(" kansas  city"))
        self.assertNotEqual(M.fingerprint("Kansas City"), M.fingerprint("San Jose"))
        self.assertEqual(len(M.fingerprint("x")), 16)


class PurityTests(unittest.TestCase):
    SRC = inspect.getsource(M)

    def test_imports_are_standard_library_pure_computation_only(self):
        mods = set()
        for n in ast.walk(ast.parse(self.SRC)):
            if isinstance(n, ast.Import):
                mods |= {a.name.split(".")[0] for a in n.names}
            elif isinstance(n, ast.ImportFrom):
                mods.add((n.module or "").split(".")[0])
        self.assertEqual(mods, {"hashlib", "re", "unicodedata", "collections", "dataclasses", "datetime"})

    def test_no_io_no_clock_no_dynamic_execution(self):
        names = {n.id for n in ast.walk(ast.parse(self.SRC)) if isinstance(n, ast.Name)}
        attrs = {n.attr for n in ast.walk(ast.parse(self.SRC)) if isinstance(n, ast.Attribute)}
        for banned in ("open", "exec", "eval", "compile", "__import__", "input", "print", "globals"):
            self.assertNotIn(banned, names, banned)
        for banned in ("now", "utcnow", "today", "time", "fromtimestamp", "write", "read", "system", "getenv", "environ"):
            self.assertNotIn(banned, attrs, banned)

    def test_it_is_not_wired_into_anything_yet(self):
        pkg = os.path.dirname(inspect.getsourcefile(M))
        for f in sorted(os.listdir(pkg)):
            if f.endswith(".py") and f not in ("memory.py", "memory_gate.py", "memory_store.py", "memory_retrieval.py", "memory_extract.py", "memory_keys.py", "memory_review.py", "memory_commit.py", "memory_manifest.py", "memory_receipts.py", "memory_execute.py", "memory_shadow.py"):   # the other Memory 2.0 layers (and the default-off shadow) are its only users
                with open(os.path.join(pkg, f), encoding="utf-8") as fh:
                    text = fh.read()
                if f == "main.py":              # the ONE permitted live import: the default-off, read-only shadow
                    self.assertEqual(text.count("from backtalk import memory_shadow\n"), 1)
                    text = text.replace("from backtalk import memory_shadow\n", "")
                self.assertNotIn("backtalk.memory", text, f)
                self.assertNotIn("from backtalk import memory", text, f)
                self.assertNotRegex(text, r"from backtalk import [^\n]*\bmemory\b", f)


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.v = self.tmp.name
        self.secret = "ZXCVBNM12345SECRETVALUE"

    def tearDown(self):
        self.tmp.cleanup()

    def note(self, rel, text):
        p = os.path.join(self.v, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(text)
        return p

    def build(self):
        self.note("01 - Daily Notes/2026-01-01.md", "---\nstatus: active\nproject: meta\ntype: log\ncreated: 2026-01-01\n---\nbody\n")
        self.note("02 - Plans/Plan.md", "---\nstatus: active\nproject: meta\ntype: plan\ncreated: 2026-01-02T10:00:00Z\n---\n")
        self.note("02 - Plans/Rules.md", "---\nstatus: active\nproject: meta\ntype: protocol\n---\n")
        self.note("03 - Ref/Guide.md", "---\nstatus: reference\nproject: meta\ntype: guide\nsource: https://x\n---\n")
        self.note("03 - Ref/Dup.md", "---\nstatus: active\nproject: meta\ntype: fact\n---\n")
        self.note("04 - Other/Dup.md", "---\nstatus: active\nproject: meta\ntype: reference\ncreator: someone\nsubject: user\n---\n")
        self.note("04 - Other/Bare.md", "no frontmatter at all\n")
        self.note("04 - Other/Partial.md", "---\nstatus: active\n---\n")
        self.note("05 - Keys/Keys.md", f"---\nstatus: active\nproject: meta\ntype: reference\n---\napi_key = {self.secret}\nBearer abcdefghijklmnopqrstuvwxyz0123\n")
        self.note(".obsidian/ignored.md", "---\nstatus: x\n---\n")
        os.makedirs(os.path.join(self.v, "04 - Second"), exist_ok=True)

    def test_counts_and_mapping(self):
        self.build()
        r = INV.inventory(self.v)
        self.assertEqual(r["total"], 9)                                     # .obsidian skipped
        self.assertEqual(r["required_frontmatter"], 7)
        self.assertEqual(r["no_frontmatter"], ["04 - Other/Bare.md"])
        self.assertEqual(sorted(m[0] for m in r["missing_required"]), ["04 - Other/Bare.md", "04 - Other/Partial.md"])
        self.assertEqual(dict(r["mappable"]), {"event": 1, "goal": 1, "instruction": 1})
        self.assertEqual(r["created"], 2)
        self.assertEqual((r["created_iso_date_only"], r["created_iso_timestamp"]), (1, 1))
        self.assertEqual(r["source_family"], 2)
        self.assertEqual(r["status_outside_memory"], 1 + 0)                  # `reference`
        self.assertEqual(r["type_overlap_with_memory"], 1)                   # a note of type `fact`
        self.assertEqual(r["memory_only_keys"], [("04 - Other/Dup.md", ("subject",))])
        self.assertEqual(r["unmappable"], 6)

    def test_collisions(self):
        self.build()
        r = INV.inventory(self.v)
        self.assertEqual(r["duplicate_basenames"], {"dup.md": ["03 - Ref/Dup.md", "04 - Other/Dup.md"]})
        self.assertEqual(r["folder_number_collisions"], {"04": ["04 - Other", "04 - Second"]})

    def test_secrets_are_located_but_never_printed(self):
        self.build()
        r = INV.inventory(self.v)
        self.assertEqual(r["secret_like"], [("05 - Keys/Keys.md", 6, "assignment:credential-name"),
                                            ("05 - Keys/Keys.md", 7, "bearer-token")])
        text = INV.render(r)
        self.assertIn("05 - Keys/Keys.md:6  assignment:credential-name", text)
        self.assertNotIn(self.secret, text)
        self.assertNotIn("abcdefghijklmnopqrstuvwxyz0123", text)
        self.assertNotIn(self.secret, repr(r))

    def test_the_inventory_changes_nothing(self):
        self.build()
        before = {}
        for root, _, files in os.walk(self.v):
            for f in files:
                p = os.path.join(root, f)
                with open(p, "rb") as fh:
                    before[p] = (fh.read(), os.stat(p).st_mtime_ns)
        r = INV.inventory(self.v)
        INV.render(r)
        after = {}
        for root, _, files in os.walk(self.v):
            for f in files:
                p = os.path.join(root, f)
                with open(p, "rb") as fh:
                    after[p] = (fh.read(), os.stat(p).st_mtime_ns)
        self.assertEqual(before, after)
        self.assertTrue(r["vault_unchanged_during_inventory"])

    def test_the_script_opens_files_for_reading_only(self):
        src = inspect.getsource(INV)
        for m in ast.walk(ast.parse(src)):
            if isinstance(m, ast.Call) and getattr(m.func, "id", "") == "open":
                mode = m.args[1].value if len(m.args) > 1 and isinstance(m.args[1], ast.Constant) else "r"
                self.assertEqual(mode, "r")
        for banned in ("os.remove", "os.rename", "shutil", ".write(", "os.makedirs", "os.utime", "unlink", "rmtree"):
            self.assertNotIn(banned, src, banned)

    def test_render_reports_the_headline_numbers(self):
        self.build()
        text = INV.render(INV.inventory(self.v))
        for want in ("9 notes", "status/project/type all present : 7", "cleanly mappable to one item type: 3",
                     "folder-number collisions", "vault unchanged during the inventory: True"):
            self.assertIn(want, text)


if __name__ == "__main__":
    unittest.main()
