"""Memory 2.0, Phase 4 hardening: tests for the gaps the Phase 4 mutation run found in the Phase 1-3 modules
(backtalk/memory.py, memory_gate.py, memory_store.py). Each test says which behavior was unpinned. They change
no code; they only pin what the code already does. Everything runs on temporary stores."""
import json
import os
import unittest
from dataclasses import replace
from datetime import datetime, tzinfo
from unittest import mock

from backtalk import memory as M
from backtalk import memory_gate as G
from backtalk import memory_store as S
from tests.test_memory_store import StoreCase, EV, NOW, day, item, mid, old_item, raw


class NoOffset(tzinfo):
    """A timezone object that reports no offset: Python treats such a datetime as naive."""
    def utcoffset(self, dt):
        return None


def proposal(n=1, ev="I live in San Jose now", **over):
    return G.Proposal(item(n, **over), ev)


# ---- Phase 1: the core --------------------------------------------------------------------------

class CoreGapTests(unittest.TestCase):
    def test_a_timezone_that_reports_no_offset_is_naive(self):
        with self.assertRaises(ValueError) as cm:
            M.is_active(item(), datetime(2026, 2, 1, tzinfo=NoOffset()))
        self.assertEqual(str(cm.exception), "time must be timezone-aware")

    def test_the_bad_time_message_is_fixed_text(self):
        with self.assertRaises(ValueError) as cm:
            M.is_active(item(), "tomorrow")
        self.assertEqual(str(cm.exception), "time must be YYYY-MM-DDTHH:MM:SSZ or an aware datetime")

    def test_normalize_value_collapses_whitespace_to_single_spaces(self):
        self.assertEqual(M.normalize_value("  Hello\tWORLD \n x "), "hello world x")

    def test_ids_and_fingerprints_are_stable_across_releases(self):
        self.assertEqual(M.derive_id("user", "residence", "San Jose", "2026-01-10T00:00:00Z"), "mem_aa108de93b7d")
        self.assertEqual(M.fingerprint("San  Jose"), "8ee3459def4996c3")
        self.assertEqual(M.fingerprint("san jose"), "8ee3459def4996c3")

    def test_the_relation_names_are_fixed(self):
        self.assertEqual((M.EXACT_DUPLICATE, M.SAME_KEY_DIFFERENT_VALUE, M.DISTINCT),
                         ("exact_duplicate", "same_key_different_value", "distinct"))
        a, b = item(1), item(2, value="Austin")
        self.assertEqual((M.classify_pair(a, item(3)), M.classify_pair(a, b), M.classify_pair(a, item(4, attribute="language"))),
                         ("exact_duplicate", "same_key_different_value", "distinct"))

    def test_validate_names_a_blank_id_and_a_missing_source(self):
        self.assertEqual(M.validate(replace(item(), id="")), ("missing:id",))
        self.assertEqual(M.validate(replace(item(), source=None)), ("missing:source",))

    def test_a_source_reference_may_not_contain_a_carriage_return(self):
        self.assertEqual(M.validate(replace(item(), source=M.Source("user", "turn\r41"))), ("invalid:source",))
        self.assertEqual(M.validate(replace(item(), source=M.Source("user", "turn\n41"))), ("invalid:source",))

    def test_parse_refuses_an_object_that_has_keys_but_no_get(self):
        class KeysOnly:
            def keys(self):
                return []
        self.assertEqual(M.parse(KeysOnly()), (None, ("invalid:not_a_mapping",)))

    def test_parse_reports_a_bad_supersedes_next_to_other_errors(self):
        got, errs = M.parse({**raw(), "supersedes": [1], "bogus": 1})
        self.assertIsNone(got)
        self.assertEqual(errs, ("unknown_field:bogus", "invalid:supersedes"))

    def test_is_active_answers_with_a_real_false(self):
        self.assertIs(M.is_active(replace(item(), value=""), NOW), False)
        self.assertIs(M.is_active(item(status="archived"), NOW), False)
        self.assertIs(M.is_active(item(status="archived"), NOW, include_archived=True), True)

    def test_a_weaker_item_that_ties_the_leader_exactly_is_shadowed_not_superseded(self):
        stated = item(1)
        inferred = item(2, confidence="inferred", value="Austin", source={"kind": "session", "ref": "s"})
        res = M.resolve([stated, inferred], NOW)
        self.assertEqual([(x.id, why) for x, why in res.history], [(mid(2), "shadowed")])

    def test_a_supersession_chain_that_dangles_is_rejected_not_a_crash(self):
        a = item(1, supersedes=[mid(2)], value="Austin", valid_from=day(20), created=day(20), updated=day(20))
        b = item(2, supersedes=[mid(9)], valid_from=day(10))
        res = M.resolve([a, b], NOW)
        self.assertEqual(dict(res.rejected), {mid(2): (f"supersedes:unknown:{mid(9)}",),
                                              mid(1): (f"supersedes:unknown:{mid(2)}",)})

    def test_reaches_follows_links_and_answers_with_a_real_false(self):
        by_id = {mid(1): item(1, supersedes=[mid(2)]), mid(2): item(2, supersedes=[mid(1)])}
        self.assertIs(M._reaches(by_id, mid(1), mid(1)), True)
        self.assertIs(M._reaches({}, mid(1), mid(2)), False)
        self.assertIs(M._reaches(by_id, mid(1), mid(3)), False)

    def test_the_value_objects_are_frozen(self):
        with self.assertRaises(Exception):
            M.resolve([], NOW).active = ()
        with self.assertRaises(Exception):
            M.Conflict("user.residence", ()).key = "x"


# ---- Phase 2: the gate --------------------------------------------------------------------------

def base_words(n):
    """Evidence text of exactly n characters made of plain words."""
    ev = "I live in San Jose now"
    return ev + (" ok" * 200)[:n - len(ev)]


class GateGapTests(unittest.TestCase):
    def decide(self, p, now=NOW):
        return G.evaluate(p, [], now)

    def test_evidence_of_exactly_three_hundred_characters_is_accepted_and_one_more_is_not(self):
        self.assertEqual(len(base_words(300)), 300)
        self.assertEqual(self.decide(proposal(ev=base_words(300))).decision, G.ACCEPT)
        d = self.decide(proposal(ev=base_words(301)))
        self.assertEqual((d.decision, d.reason), (G.REJECT, "invalid_evidence"))

    def inferred(self, ev):
        return self.decide(proposal(ev=ev, confidence="inferred", source={"kind": "session", "ref": "s1"}))

    def test_inferred_evidence_must_be_at_least_24_characters(self):
        self.assertEqual((len("he buys tea at the cafes"), len("he buys tea at the cafe")), (24, 23))
        self.assertEqual(self.inferred("he buys tea at the cafes").decision, G.ACCEPT)
        self.assertEqual(self.inferred("he buys tea at the cafe").reason, "weak_evidence")

    def test_inferred_evidence_must_have_at_least_five_words(self):
        self.assertEqual(self.inferred("alpha bravo charlie delta echo").decision, G.ACCEPT)
        self.assertEqual(self.inferred("alpha bravo charlie delta").reason, "weak_evidence")

    def test_inferred_evidence_must_have_at_least_three_significant_words(self):
        self.assertEqual(self.inferred("he buys tea now so we go to it").decision, G.ACCEPT)       # buys, tea, now
        self.assertEqual(self.inferred("he buys tea so we go to it").reason, "weak_evidence")      # buys, tea

    def test_a_three_letter_value_and_a_value_with_a_digit_are_significant(self):
        self.assertEqual(self.decide(proposal(value="tea", ev="I drink tea daily")).decision, G.ACCEPT)
        self.assertEqual(self.decide(proposal(value="b2", ev="my seat is b2 today")).decision, G.ACCEPT)
        self.assertEqual(self.decide(proposal(value="tea", ev="I drink coffee daily")).reason, "evidence_does_not_support_value")

    def test_evidence_that_is_exactly_the_value_supports_it(self):
        d = self.decide(proposal(ev="San Jose"))
        self.assertEqual((d.decision, d.reason), (G.ACCEPT, "accepted"))

    def test_a_proposal_whose_item_has_a_non_text_id_is_rejected_not_a_crash(self):
        d = self.decide(G.Proposal(replace(item(), id=5), EV))
        self.assertEqual((d.decision, d.reason, d.memory_id), (G.REJECT, "invalid_item", None))

    def test_parse_proposal_refuses_an_object_that_has_keys_but_no_get(self):
        class KeysOnly:
            def keys(self):
                return []
        self.assertEqual(G.parse_proposal(KeysOnly()), (None, ("invalid:not_a_mapping",)))

    def test_a_timezone_that_reports_no_offset_is_naive(self):
        with self.assertRaises(ValueError) as cm:
            G.evaluate(proposal(), [], datetime(2026, 2, 1, tzinfo=NoOffset()))
        self.assertEqual(str(cm.exception), "time must be timezone-aware")
        with self.assertRaises(ValueError) as cm:
            G.evaluate(proposal(), [], "soon")
        self.assertEqual(str(cm.exception), "time must be YYYY-MM-DDTHH:MM:SSZ or an aware datetime")

    def test_a_proposal_is_immutable(self):
        with self.assertRaises(Exception):
            proposal().evidence = "changed"

    def test_every_live_state_name_is_refused_as_temporary(self):
        self.assertEqual(len(G.TEMPORARY_NAMES), 33)
        for name in sorted(G.TEMPORARY_NAMES):
            for field in ("attribute", "subject"):
                d = self.decide(proposal(**{field: name}))
                self.assertEqual((d.decision, d.reason), (G.REJECT, "temporary_context"), f"{field}={name}")

    def test_every_filler_word_alone_is_low_value(self):
        self.assertEqual(len(G.FILLER_WORDS), 66)
        for w in sorted(G.FILLER_WORDS):
            ev = " ".join([w] * 8)
            d = self.decide(proposal(value="Austin", ev=ev))
            self.assertEqual((d.decision, d.reason), (G.REJECT, "low_value"), w)

    CREDENTIAL_SAMPLES = {
        "credential:url": "postgres://admin:hunter22@db.example.com/app",
        "credential:bearer": "Bearer abcdefghijkl1234",
        "credential:header": "Authorization: abcdef",
        "credential:assignment": "my password is hunter2hunter2",
        "credential:provider-key": "sk-abcdefghijklmnop1234",
        "credential:github-token": "ghp_abcdefghijklmnopqrstuvwx",
        "credential:slack-token": "xoxb-1234567890-abcdef",
        "credential:aws-key": "AKIAABCDEFGHIJKLMNOP",
        "credential:jwt": "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abc",
        "credential:private-key": "-----BEGIN RSA PRIVATE KEY-----",
        "credential:card-number": "4111 1111 1111 1111",
        "credential:ssn": "123-45-6789",
        "credential:opaque-blob": "abcdefghijklmnopqrstuvwxyz012345",
    }

    def test_every_credential_category_has_a_sample_that_names_it(self):
        self.assertEqual(set(self.CREDENTIAL_SAMPLES), {label for label, _ in G._CREDENTIAL_PATTERNS})
        for label, sample in self.CREDENTIAL_SAMPLES.items():
            self.assertIn(label, G.screen_secrets(sample), label)

    def test_the_credential_categories_come_back_in_a_fixed_order(self):
        text = " ".join(self.CREDENTIAL_SAMPLES.values())
        self.assertEqual(G.screen_secrets(text), tuple(label for label, _ in G._CREDENTIAL_PATTERNS))


# ---- Phase 3: the store -------------------------------------------------------------------------

class StoreGapTests(StoreCase):
    def write_raw(self, name, text):
        os.makedirs(self.memdir, exist_ok=True)
        with open(os.path.join(self.memdir, name), "w", encoding="utf-8", newline="") as f:
            f.write(text)

    def write_item(self, it, ev=EV):
        self.write_raw(it.id + ".md", S.serialize(it, ev))

    def test_a_store_error_prints_its_reason_then_its_details(self):
        self.assertEqual(str(S.StoreError("x")), "x")
        self.assertEqual(str(S.StoreError("x", ("a", 2))), "x: a, 2")
        e = S.StoreError("y", ["p"])
        self.assertEqual((e.reason, e.details), ("y", ("p",)))

    def test_stored_and_applied_are_immutable(self):
        for obj, name in ((S.Stored(item(), EV), "evidence"), (S.Applied(mid(1), ()), "memory_id")):
            with self.assertRaises(Exception):
                setattr(obj, name, "x")

    def test_the_json_writer_refuses_not_a_number(self):
        with self.assertRaises(ValueError):
            S._js(float("nan"))

    def test_serialize_names_its_refusals(self):
        self.refused("invalid_item", S.serialize, replace(item(), value=""), EV)
        self.refused("invalid_evidence", S.serialize, item(), "two\nlines")

    def test_parse_file_names_its_layout_failures(self):
        good = S.serialize(item(), EV)
        self.assertEqual(self.refused("bad_format", S.parse_file, "no newline").details, ("ending",))
        self.assertEqual(self.refused("bad_format", S.parse_file, "---\n").details, ("layout",))
        self.assertEqual(self.refused("bad_format", S.parse_file, good.replace("memory_schema: 1", "memory_schema: 2")).details,
                         ("schema",))
        self.assertEqual(self.refused("bad_format", S.parse_file, good + "extra\n").details, ("layout",))

    def test_a_file_that_stops_right_after_the_field_lines_is_a_layout_error_not_a_crash(self):
        good = S.serialize(item(), EV)
        lines = good[:-1].split("\n")
        self.assertEqual(len(lines), len(S.FIELD_ORDER) + 6)
        for keep in range(0, len(lines)):
            text = "\n".join(lines[:keep]) + "\n"
            self.assertEqual(self.refused("bad_format", S.parse_file, text).details, ("layout",), keep)

    def test_a_stray_file_is_one_problem_and_is_not_parsed_as_well(self):
        self.put(item(1))
        self.write_raw("README.md", "hello with no final newline")
        self.write_raw(mid(2) + ".txt", S.serialize(item(2), EV))
        self.assertEqual(self.store.verify(), ("unexpected_file:README.md", f"unexpected_file:{mid(2)}.txt"))

    def test_put_returns_the_id(self):
        self.assertEqual(self.put(item(5)), mid(5))

    def test_an_unreadable_file_is_one_problem_and_does_not_borrow_the_previous_files_identity(self):
        self.put(item(1))
        self.write_raw(mid(2) + ".md", "")
        with open(os.path.join(self.memdir, mid(2) + ".md"), "wb") as f:
            f.write(b"\xff\xfe\x00bad")
        self.assertEqual(self.store.verify(), (f"{mid(2)}.md:unreadable",))

    def test_problem_codes_carry_their_details_joined_by_commas(self):
        self.put(item(1))
        text = S.serialize(item(2), EV).replace('value: "San Jose"', 'value: ""').replace('"stated"', '"weird"')
        self.write_raw(mid(2) + ".md", text)
        blank = S.serialize(item(3), EV)
        self.write_raw(mid(3) + ".md", blank[:blank.index("## Evidence")] + "## Evidence\n\n\n")
        self.assertEqual(self.store.verify(), (
            f"{mid(2)}.md:invalid_item:missing:value,invalid:confidence",
            f"{mid(3)}.md:invalid_evidence"))

    def test_records_the_resolver_rejects_are_listed_with_all_their_codes(self):
        self.write_item(item(1, supersedes=[mid(8), mid(9)]))
        self.assertEqual(self.store.verify(), (f"{mid(1)}:supersedes:unknown:{mid(8)},supersedes:unknown:{mid(9)}",))

    def test_a_clock_with_no_offset_is_invalid_time(self):
        self.refused("invalid_time", S._stamp, datetime(2026, 2, 1, tzinfo=NoOffset()))
        self.refused("invalid_time", S._stamp, datetime(2026, 2, 1))
        self.assertEqual(S._stamp("2026-02-01T00:00:00Z"), "2026-02-01T00:00:00Z")

    def test_the_temp_file_is_a_dot_tmp_part_file_in_the_memory_folder(self):
        seen = []

        def boom(src, dst):
            seen.append((src, dst))
            raise OSError("disk")
        with mock.patch.object(S, "_replace", boom):
            self.refused("write_failed", self.put, item())
        (src, dst), = seen
        self.assertEqual(os.path.dirname(src), self.memdir)
        self.assertTrue(os.path.basename(src).startswith(".tmp-") and src.endswith(".part"), src)
        self.assertFalse(os.path.exists(src))

    def test_the_lock_file_is_readable_by_all_and_writable_by_the_owner_only(self):
        old = os.umask(0o022)
        self.addCleanup(os.umask, old)
        self.put(item())
        self.assertEqual(os.stat(os.path.join(self.memdir, ".lock")).st_mode & 0o777, 0o644)

    def test_the_lock_file_is_not_a_problem(self):
        self.put(item())
        self.assertTrue(os.path.exists(os.path.join(self.memdir, ".lock")))
        self.assertEqual(self.store.verify(), ())

    def accept(self, n=1, **over):
        prop = G.Proposal(item(n, **over), EV)
        return prop, G.Decision(G.ACCEPT, "accepted", memory_id=mid(n), evidence=EV, confidence="stated", sensitivity="normal")

    def test_a_decision_that_would_create_a_conflict_is_refused_after_the_fact(self):
        self.put(item(1))
        prop, decision = self.accept(2, value="Austin")
        with mock.patch.object(G, "evaluate", return_value=decision):
            e = self.refused("postcondition_failed", self.store.apply, decision, prop, NOW)
        self.assertEqual(e.details, ("conflict",))
        self.assertEqual([s.item.id for s in self.store.list_items()], [mid(1)])

    def test_a_conflict_that_does_not_involve_the_new_memory_is_not_the_reason_to_refuse(self):
        self.write_item(item(1))
        self.write_item(item(2, value="Austin"))
        prop, decision = self.accept(3, value="Boise", valid_from=day(2), created=day(2), updated=day(2))
        with mock.patch.object(G, "evaluate", return_value=decision):
            e = self.refused("postcondition_failed", self.store.apply, decision, prop, NOW)
        self.assertEqual(e.details, ("new_item_not_in_force",))
        self.assertEqual(len(os.listdir(self.memdir)), 3)                  # two items and the lock: nothing was written

    def test_a_supersession_that_the_resolver_rejects_is_refused_with_every_code(self):
        self.put(item(1, attribute="language", value="Spanish"), "I speak Spanish")
        self.put(item(2, attribute="vehicle", value="Tesla"), "I drive a Tesla")
        prop, _ = self.accept(3, value="Austin", valid_from=day(15), created=day(15), updated=day(15))
        decision = G.Decision(G.ACCEPT, "accepted_supersedes", memory_id=mid(3), evidence=EV, confidence="stated",
                              sensitivity="normal", supersedes=(mid(1), mid(2)), effective_from=day(15))
        with mock.patch.object(G, "evaluate", return_value=decision):
            e = self.refused("postcondition_failed", self.store.apply, decision, prop, NOW)
        self.assertEqual(e.details, (f"{mid(3)}:supersedes:key_mismatch:{mid(1)},supersedes:key_mismatch:{mid(2)}",))


if __name__ == "__main__":
    unittest.main()
