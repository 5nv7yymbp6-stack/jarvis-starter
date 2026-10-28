"""Memory 2.0, Phase 6: the scanner and the evaluation harness (backtalk/evals/memory_extraction), pinned exactly, because
the numbers they report are the Phase 6 results. Synthetic fixtures only."""
import contextlib
import io
import json
import os
import unittest
from collections import Counter
from dataclasses import replace
from unittest import mock

from backtalk import memory_extract as X
from backtalk.evals.memory_extraction import evaluate as E
from backtalk.evals.memory_extraction import scan
from tests.test_memory_extract import NOW, Base, cands_of, entry, note, read, turn, write


class ScanReadingTests(Base):
    def lines(self):
        return [entry("I prefer short answers when we talk.", uuid="t1"),
                entry("I prefer calm answers when we talk.", ep="sdk-py", uuid="v1"),
                {"type": "assistant", "message": {"content": "x"}},
                "{broken", "[1]",
                entry("x", uuid="tr", toolUseResult={"a": 1}),
                entry("x", uuid="m", isMeta=True),
                entry("", uuid="e")]

    def test_the_counters_are_exact(self):
        tr = self.mkdir("tr")
        self.write_transcript(tr, "a.jsonl", self.lines())
        turns, st = scan.read_turns(tr)
        self.assertEqual(dict(st), {"files": 1, "lines": 8, "malformed_lines": 2, "user_entries": 5, "turn_typed": 1,
                                     "turn_voice": 1, "excluded:tool_result": 1, "excluded:meta": 1, "excluded:empty_after_clean": 1})
        self.assertEqual([(t.channel, t.line, t.file) for t in turns], [("typed", 1, "a.jsonl"), ("voice", 2, "a.jsonl")])

    def test_a_line_is_oversize_only_when_longer_than_the_limit(self):
        tr = self.mkdir("tr")
        line = json.dumps(entry("I prefer short answers when we talk.", uuid="t1"))
        self.write_transcript(tr, "a.jsonl", [line])
        size = len((line + "\n").encode())
        with mock.patch.object(scan, "MAX_LINE_BYTES", size):
            self.assertEqual(len(scan.read_turns(tr)[0]), 1)
        with mock.patch.object(scan, "MAX_LINE_BYTES", size - 1):
            turns, st = scan.read_turns(tr)
        self.assertEqual((len(turns), st["oversize_lines"]), (0, 1))

    def test_an_unknown_exclusion_is_still_counted(self):
        tr = self.mkdir("tr")
        self.write_transcript(tr, "a.jsonl", [entry("x")])
        with mock.patch.object(X, "classify_turn", return_value=(None, None)):
            self.assertEqual(scan.read_turns(tr)[1]["excluded:unknown"], 1)

    def test_only_eligible_notes_are_loaded_with_slash_paths(self):
        v = self.mkdir("vault")
        keep = ["VAULT-INDEX.md", "LESSONS.md", "BOOT-CONFIG.md", "Active Priorities.md", "02 - Wholesaling/Buy Box.md",
                "03 - Markets/Kansas City (Missouri).md", "01 - Daily Notes/09 - September 2026/2026-09-24.md"]
        skip = ["01 - Daily Notes/Daily Note Template.md", "01 - Daily Notes/09 - September 2026/notes.md", "Other.md",
                "09 - Resources/Phone Bridge.md", "02 - Wholesaling/Seller Leads.md"]
        for rel in keep + skip:
            self.write_note(v, rel, f"text of {rel}\n")
        notes = scan.load_notes(v)
        self.assertEqual([n.path for n in notes], sorted(keep))
        self.assertEqual(notes[0].sha, X.sha(notes[0].text))
        os.symlink(os.path.join(v, "Other.md"), os.path.join(v, "01 - Daily Notes", "09 - September 2026", "2026-09-25.md"))
        self.assertNotIn("2026-09-25", " ".join(n.path for n in scan.load_notes(v)))
        self.assertEqual(scan.load_notes(os.path.join(self.root, "nowhere")), [])

    def test_the_probe_paths_are_the_known_conflict_notes(self):
        self.assertEqual(scan.probe_paths(), {"04 - Wholesaling/04 - Wholesaling.md", "Active Priorities.md",
                                              "03 - Markets/Kansas City (Missouri).md", "01 - Daily Notes/09 - September 2026/2026-09-14.md",
                                              "VAULT-INDEX.md", "BOOT-CONFIG.md", "01 - Daily Notes/09 - September 2026/2026-09-30.md",
                                              "02 - Wholesaling/Buy Box.md", "03 - Markets/Markets.md"})

    def test_the_defaults(self):
        self.assertTrue(scan.DEFAULT_OUT.endswith("/.backtalk-memory-review"))
        self.assertTrue(scan.DEFAULT_TRANSCRIPTS.endswith("/.claude/projects/-Users-yousif-Desktop-jarvis"))
        self.assertEqual(scan.DEFAULT_VAULT, "/Users/yousif/Desktop/jarvis/jarvis")
        self.assertEqual(scan.MAX_LINE_BYTES, 4_000_000)
        self.assertEqual(scan.EXTRACT_NOTES, ("VAULT-INDEX.md", "03 - Markets/Kansas City (Missouri).md",
                                              "04 - Wholesaling/04 - Wholesaling.md", "10 - New Ventures/10 - New Ventures.md",
                                              "00 - Inbox/Approved Memory Statements.md"))      # deliberate: the append-only approved-statements note
        self.assertEqual(scan.QUOTE_NOTES, ("LESSONS.md", "BOOT-CONFIG.md", "VAULT-INDEX.md"))
        self.assertEqual(scan.PACK_FILES, ("review_pack.json", "report.md", "private_review.json"))


class ScanRunTests(Base):
    def fixture(self):
        v, tr, out = self.mkdir("repo", "vault"), self.mkdir("tr"), self.mkdir("out")
        self.write_note(v, "VAULT-INDEX.md", "## What I Want\n\n- A cash buyer list deep enough in each fund's market.\n")
        self.write_note(v, "LESSONS.md", "Never answer a problem by telling me to go ask someone or something else.\n")
        self.write_transcript(tr, "aaaaaaaa-0000.jsonl", [
            entry("I prefer short answers when we talk.", uuid="t1"),
            entry("Never answer a problem by telling me to go ask someone or something else.", uuid="t2"),
            entry("I prefer calm answers when we talk.", ep="sdk-py", uuid="v1")])
        return v, tr, out

    def test_the_meta_block_is_exact(self):
        v, tr, out = self.fixture()
        cands, conflicts, meta, notes = scan.run_pipeline(tr, v, NOW)
        self.assertEqual(sorted(meta), ["channels", "exclusions", "generated_at", "notes_fingerprint", "notes_read", "transcript_files"])
        self.assertEqual((meta["generated_at"], meta["transcript_files"], meta["notes_read"], meta["channels"]),
                         (NOW, 1, 2, {"typed": 2, "voice": 1}))
        self.assertEqual(meta["notes_fingerprint"], X.sha("".join(n.path + n.sha for n in notes)))
        self.assertEqual(meta["exclusions"]["candidates"], {"quoted_from_vault": 1})
        self.assertEqual(meta["exclusions"]["turns"]["turn_typed"], 2)
        self.assertEqual(len(cands), 3)
        self.assertEqual(len([k for k in conflicts if k.kind == "known"]), 7)

    def test_the_run_folder_is_named_from_the_timestamp_and_can_be_rewritten(self):
        v, tr, out = self.fixture()
        cands, conflicts, meta, _ = scan.run_pipeline(tr, v, NOW)
        done = scan.write_pack(out, cands, conflicts, meta, forbidden=[v, tr])
        self.assertEqual(os.path.basename(done["dir"]), "run-20260201000000")
        again = scan.write_pack(out, cands, conflicts, meta, forbidden=[v, tr])
        self.assertEqual(done, again)
        self.assertEqual(sorted(os.listdir(done["dir"])), ["private_review.json", "report.md", "review_pack.json"])

    def test_a_failed_write_leaves_no_temp_file_and_no_partial_pack(self):
        d = self.mkdir("dest")
        with mock.patch.object(os, "replace", side_effect=OSError("disk")):
            with self.assertRaises(OSError):
                scan._write_private(os.path.join(d, "x.json"), "text")
        self.assertEqual(os.listdir(d), [])
        scan._write_private(os.path.join(d, "x.json"), "text")
        self.assertEqual(os.listdir(d), ["x.json"])
        with mock.patch.object(os, "chmod", side_effect=OSError("denied")):
            with self.assertRaises(OSError):
                scan._write_private(os.path.join(d, "y.json"), "text")
        self.assertEqual(os.listdir(d), ["x.json"])

    def test_the_refusal_message_is_fixed(self):
        with self.assertRaises(ValueError) as cm:
            scan.output_dir_ok(self.mkdir("vault"), forbidden=[os.path.join(self.root, "vault")])
        self.assertEqual(str(cm.exception), "the review pack must be written outside the vault, the repository, the transcripts and the store")
        self.assertEqual(scan.output_dir_ok(self.mkdir("elsewhere"), forbidden=[os.path.join(self.root, "vault")]),
                         os.path.realpath(self.mkdir("elsewhere")))

    def run_main(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            scan.main(list(argv))
        return out.getvalue()

    def test_the_command_line(self):
        v, tr, out = self.fixture()
        printed = self.run_main("--vault", v, "--transcripts", tr, "--out", out, "--now", NOW)
        lines = printed.strip().split("\n")
        self.assertRegex(lines[0], r"^wrote \['private_review.json', 'report.md', 'review_pack.json'\] to .*/run-20260201000000$")
        self.assertRegex(lines[1], r"^candidates 3  gate \{.*\}  confirm 2$")
        run_dir = os.path.join(out, "run-20260201000000")
        self.assertEqual(sorted(os.listdir(run_dir)), ["private_review.json", "report.md", "review_pack.json"])
        out2 = self.mkdir("out2")
        printed = self.run_main("--vault", v, "--transcripts", tr, "--out", out2, "--now", NOW, "--no-private")
        self.assertIn("['report.md', 'review_pack.json']", printed)

    def test_a_bad_now_exits_with_a_message(self):
        v, tr, out = self.fixture()
        with self.assertRaises(SystemExit) as cm:
            scan.main(["--vault", v, "--transcripts", tr, "--out", out, "--now", "soon"])
        self.assertEqual(str(cm.exception), "--now must be YYYY-MM-DDTHH:MM:SSZ")

    def test_the_command_line_refuses_an_output_inside_the_vault(self):
        v, tr, out = self.fixture()
        with self.assertRaises(ValueError):
            scan.main(["--vault", v, "--transcripts", tr, "--out", os.path.join(v, "pack"), "--now", NOW])
        self.assertFalse(os.path.exists(os.path.join(v, "pack")))


class EvaluateMetricTests(Base):
    def cand(self, **over):
        (c,), _ = cands_of([turn("I prefer short answers when we talk.")])
        return replace(c, **over)

    def test_gold_recall_rules(self):
        c1 = self.cand(attribute="residence", where="a.md", origin="vault", source_kind="note", confidence="inferred", corroboration=None)
        c2 = self.cand(attribute="goal_x", where="b.md")
        gold = {"positives": [
            {"id": "k_ok", "key": "residence", "path": "a.md"},
            {"id": "k_wrong_key", "key": "occupation", "path": "a.md"},
            {"id": "k_wrong_path", "key": "residence", "path": "z.md"},
            {"id": "p_any", "key_prefix": ["nope_", "goal_"], "path": "b.md", "min_count": 1},
            {"id": "p_path", "key_prefix": ["goal_"], "path": "z.md", "min_count": 1},
            {"id": "p_two", "key_prefix": ["goal_"], "path": "b.md", "min_count": 2},
            {"id": "rx", "key": "goal_x", "path_regex": "^b\\."}], "negatives": []}
        r = E.gold_recall([c1, c2], gold)
        self.assertEqual({x["id"]: (x["found"], x["matches"]) for x in r["rows"]},
                         {"k_ok": (True, 1), "k_wrong_key": (False, 0), "k_wrong_path": (False, 0), "p_any": (True, 1),
                          "p_path": (False, 0), "p_two": (False, 1), "rx": (True, 1)})
        self.assertEqual((r["gold"], r["found"], r["recall"]), (7, 3, 3 / 7))
        self.assertEqual(E.gold_recall([], {"positives": [], "negatives": []})["recall"], None)

    def test_false_stated_rules(self):
        ok = self.cand()
        vault_stated = self.cand(origin="vault", where="a.md", id="cand_v")
        no_corr = self.cand(corroboration=None, id="cand_nc")
        voice = self.cand(corroboration="voice_independent", id="cand_vo")
        wrong_kind = self.cand(source_kind="session", id="cand_wk")
        self.assertEqual(E.false_stated([ok, voice])["structural_violations"], [])
        r = E.false_stated([ok, vault_stated, no_corr, voice, wrong_kind])
        self.assertEqual(sorted(r["structural_violations"]), ["cand_nc", "cand_v", "cand_wk"])
        self.assertEqual((r["stated"], r["rate"]), (5, 3 / 5))
        r = E.false_stated([ok, voice], {ok.id: "reject"})
        self.assertEqual((r["labelled_wrong"], r["rate"]), ([ok.id], 1 / 2))
        inferred = self.cand(confidence="inferred")
        self.assertEqual((E.false_stated([inferred])["stated"], E.false_stated([inferred])["rate"]), (0, None))

    def test_provenance_line_bounds_and_mismatches(self):
        v, tr = self.mkdir("vault"), self.mkdir("tr")
        self.write_note(v, "VAULT-INDEX.md", "## What I Want\n\n- A cash buyer list deep enough in each fund's market.\n")
        self.write_transcript(tr, "aaaaaaaa-0000.jsonl", [entry("I prefer short answers when we talk.", uuid="t1")])
        cands, *_ = scan.run_pipeline(tr, v, NOW)
        vc = next(c for c in cands if c.origin == "vault")
        tc = next(c for c in cands if c.origin == "transcript")
        self.assertEqual(E.verify_provenance(vc, tr, v), [])
        self.assertEqual(E.verify_provenance(tc, tr, v), [])
        self.assertEqual(E.verify_provenance(replace(vc, line=0), tr, v), ["line_out_of_range"])
        self.assertEqual(E.verify_provenance(replace(vc, line=5), tr, v), ["line_out_of_range"])
        self.assertEqual(E.verify_provenance(replace(vc, source_ref="vault:x#L3"), tr, v), ["ref_mismatch"])
        self.assertEqual(E.verify_provenance(replace(vc, where="missing.md"), tr, v), ["source_missing"])
        self.assertEqual(E.verify_provenance(replace(vc, text_sha="0" * 64), tr, v), ["hash_mismatch"])
        self.assertEqual(E.verify_provenance(replace(tc, channel="voice"), tr, v), ["channel_mismatch"])
        self.assertEqual(E.verify_provenance(replace(tc, source_ref="transcript:x#L1@untimed/00000000"), tr, v), ["ref_mismatch"])
        self.assertEqual(E.verify_provenance(replace(tc, line=2), tr, v), ["line_out_of_range"])
        self.assertEqual(E.verify_provenance(replace(tc, where="gone.jsonl"), tr, v), ["source_missing"])
        write(os.path.join(tr, "aaaaaaaa-0000.jsonl"), json.dumps(dict(entry("x", uuid="t1"), isMeta=True)) + "\n")
        self.assertEqual(E.verify_provenance(tc, tr, v), ["turn_no_longer_eligible"])
        write(os.path.join(tr, "aaaaaaaa-0000.jsonl"), "{broken\n")
        self.assertEqual(E.verify_provenance(tc, tr, v), ["source_missing"])

    def test_provenance_rate_counts_correct_and_checked(self):
        v, tr = self.mkdir("vault"), self.mkdir("tr")
        self.write_note(v, "VAULT-INDEX.md", "## What I Want\n\n- A cash buyer list deep enough in each fund's market.\n")
        self.write_transcript(tr, "aaaaaaaa-0000.jsonl", [entry("I prefer short answers when we talk.", uuid="t1")])
        cands, *_ = scan.run_pipeline(tr, v, NOW)
        r = E.provenance(cands + [replace(cands[0], id="cand_bad", line=99)], tr, v)
        self.assertEqual((r["checked"], r["correct"], r["rate"], list(r["problems"])), (3, 2, 2 / 3, ["cand_bad"]))
        self.assertEqual(E.provenance([], tr, v), {"checked": 0, "correct": 0, "rate": None, "problems": {}})

    def test_the_sample_default_and_precision_strata(self):
        base = self.cand()
        many = [replace(base, id=f"cand_{i:012d}") for i in range(9)]
        self.assertEqual(len(E.stratified_sample(many)), 6)
        self.assertEqual(len(E.stratified_sample(many, 2)), 2)
        labels = {many[0].id: "accept", many[1].id: "reject", many[2].id: "accept", many[3].id: "edit"}
        r = E.precision(many[:4], labels)
        self.assertEqual(r["by_stratum"], {"transcript/user": {"accept": 2, "reject": 1, "edit": 1}})
        self.assertEqual((r["sampled"], r["labelled"]), (4, 4))
        self.assertEqual(E.precision(many[:4], {many[0].id: "maybe"})["labelled"], 0)

    def test_evaluate_report_shape_and_conflict_counts(self):
        v, tr = self.mkdir("vault"), self.mkdir("tr")
        self.write_note(v, "VAULT-INDEX.md", "## What I Want\n\n- A cash buyer list deep enough in each fund's market.\n")
        self.write_note(v, "BOOT-CONFIG.md", "Double-confirm before any source-code edit\n")
        self.write_transcript(tr, "aaaaaaaa-0000.jsonl", [entry("I prefer short answers when we talk.", uuid="t1")])
        r = E.evaluate(tr, v, NOW, {"reviewer": "tester", "labels": {}})
        self.assertEqual(sorted(r), ["by_origin", "candidates", "confidence", "conflicts", "exclusions", "false_stated", "gate_decisions",
                                     "gate_reasons", "generated_at", "gold_recall", "if_affirmed", "labels_reviewer", "negative_leakage",
                                     "pack_deterministic", "precision", "provenance", "requires_confirmation", "sample_ids",
                                     "screening", "sensitivity"])
        self.assertEqual((r["generated_at"], r["candidates"], r["labels_reviewer"], r["by_origin"]), (NOW, 2, "tester", {"transcript": 1, "vault": 1}))
        self.assertEqual(r["conflicts"]["known_detected"], 0)
        self.assertEqual(r["conflicts"]["known_status"]["user.instruction_confirm_edits"], "partial")
        self.assertEqual(r["conflicts"]["generic"], [])
        self.assertEqual(r["gold_recall"]["gold"], len(E.load_gold()["positives"]))
        self.assertEqual(r["gate_decisions"], {"ACCEPT": 2})
        self.assertEqual((r["confidence"], r["sensitivity"], r["requires_confirmation"]), ({"inferred": 1, "stated": 1}, {"normal": 2}, 1))
        self.assertEqual(len(r["sample_ids"]), 2)

    def test_the_command_line_prints_json_and_can_write_it_outside_the_vault(self):
        v, tr = self.mkdir("repo", "vault"), self.mkdir("tr")
        self.write_note(v, "VAULT-INDEX.md", "## What I Want\n\n- A cash buyer list deep enough in each fund's market.\n")
        self.write_transcript(tr, "aaaaaaaa-0000.jsonl", [entry("I prefer short answers when we talk.", uuid="t1")])
        labels = os.path.join(self.root, "labels.json")
        write(labels, json.dumps({"reviewer": "t", "labels": {}}))
        dest = os.path.join(self.mkdir("res"), "e.json")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            E.main(["--vault", v, "--transcripts", tr, "--labels", labels, "--now", NOW, "--json-out", dest, "--per-stratum", "1"])
        printed = json.loads(buf.getvalue())
        self.assertEqual(printed["labels_reviewer"], "t")
        self.assertEqual(json.loads(read(dest)), printed)
        self.assertTrue(read(dest).endswith("}\n"))
        self.assertEqual(os.stat(dest).st_mode & 0o777, 0o600)
        with self.assertRaises(ValueError):
            E.main(["--vault", v, "--transcripts", tr, "--now", NOW, "--json-out", os.path.join(v, "e.json")])
        self.assertFalse(os.path.exists(os.path.join(v, "e.json")))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            E.main(["--vault", v, "--transcripts", tr, "--now", NOW])
        self.assertIsNone(json.loads(buf.getvalue())["labels_reviewer"])

    def test_the_synthetic_samples_are_each_caught_alone(self):
        def survives(text):
            e = entry(text)
            return X.classify_turn(e)[0] is not None
        for s in E.SYNTHETIC_SECRETS + E.SYNTHETIC_INJECTIONS:
            self.assertFalse(survives("I prefer short answers. " + s), s)
        for b in E.BENIGN:
            self.assertTrue(survives(b), b)
        self.assertEqual(len(E.SYNTHETIC_SECRETS), 15)
        self.assertEqual(len(E.SYNTHETIC_INJECTIONS), 5)
        r = E.screening()
        self.assertEqual((r["secrets_caught"], r["injections_caught"], r["benign"]), (15, 5, 5))


class ResidualToolTests(Base):
    def test_a_candidate_on_the_first_and_on_the_last_line_is_verifiable(self):
        v, tr = self.mkdir("vault"), self.mkdir("tr")
        self.write_note(v, "04 - Wholesaling/04 - Wholesaling.md", "Chosen market: **Columbus, Ohio** (Franklin)")
        cands, *_ = scan.run_pipeline(tr, v, NOW)
        (c,) = cands
        self.assertEqual(c.line, 1)
        self.assertEqual(E.verify_provenance(c, tr, v), [])

    def test_the_reference_fingerprint_is_checked(self):
        v, tr = self.mkdir("vault"), self.mkdir("tr")
        self.write_transcript(tr, "aaaaaaaa-0000.jsonl", [entry("I prefer short answers when we talk.", uuid="t1")])
        (c,), *_ = scan.run_pipeline(tr, v, NOW)
        ref = c.source_ref[:-8] + "00000000"
        self.assertEqual(E.verify_provenance(replace(c, source_ref=ref), tr, v), ["ref_mismatch"])

    def test_the_default_sample_is_six_per_stratum(self):
        v, tr = self.mkdir("repo", "vault"), self.mkdir("tr")
        goals = "".join(f"- Goal number {n} of the long term plan here.\n" for n in range(8))
        self.write_note(v, "VAULT-INDEX.md", "## What I Want\n\n" + goals)
        self.assertEqual(len(E.evaluate(tr, v, NOW)["sample_ids"]), 6)
        self.assertEqual(len(E.evaluate(tr, v, NOW, per_stratum=2)["sample_ids"]), 2)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            E.main(["--vault", v, "--transcripts", tr, "--now", NOW])
        self.assertEqual(len(json.loads(buf.getvalue())["sample_ids"]), 6)
        self.assertTrue(buf.getvalue().startswith('{\n  "by_origin"'))

    def test_the_default_clock_is_a_timestamp(self):
        v, tr = self.mkdir("repo", "vault"), self.mkdir("tr")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            E.main(["--vault", v, "--transcripts", tr])
        self.assertRegex(json.loads(buf.getvalue())["generated_at"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
        out = self.mkdir("out")
        with contextlib.redirect_stdout(io.StringIO()) as _:
            scan.main(["--vault", v, "--transcripts", tr, "--out", out])
        self.assertRegex(os.listdir(out)[0], r"^run-\d{14}$")

    def test_the_measured_conflict_numbers_come_from_the_real_probes(self):
        from tests.test_memory_extract import conflict_notes
        v, tr = self.mkdir("repo", "vault"), self.mkdir("tr")
        for n in conflict_notes():
            self.write_note(v, n.path, n.text)
        self.write_transcript(tr, "aaaaaaaa-0000.jsonl", [entry("I live in Springfield near the lake.", uuid="a"),
                                                          entry("I live in Shelbyville by the river.", uuid="b")])
        r = E.evaluate(tr, v, NOW, {"reviewer": "t", "labels": {}})
        self.assertEqual((r["conflicts"]["known_detected"], r["conflicts"]["known_total"]), (7, 7))
        self.assertEqual(r["conflicts"]["generic"], ["user.residence"])
        stated = [c for c in scan.run_pipeline(tr, v, NOW)[0] if c.confidence == "stated"]
        r = E.evaluate(tr, v, NOW, {"reviewer": "t", "labels": {stated[0].id: "reject"}})
        self.assertEqual(r["false_stated"]["labelled_wrong"], [stated[0].id])
        self.assertEqual(r["false_stated"]["stated"], 2)

    def test_nested_report_keys(self):
        v, tr = self.mkdir("repo", "vault"), self.mkdir("tr")
        self.write_transcript(tr, "aaaaaaaa-0000.jsonl", [entry("I prefer short answers when we talk.", uuid="t1")])
        r = E.evaluate(tr, v, NOW, {"reviewer": "t", "labels": {}})
        self.assertEqual(sorted(r["gold_recall"]), ["defined_by", "found", "gold", "recall", "rows", "transcript_gold"])
        self.assertEqual(sorted(r["gold_recall"]["rows"][0]), ["found", "id", "matches"])
        self.assertEqual(sorted(r["negative_leakage"]), ["contact_details", "leaked", "negative_sources"])
        self.assertEqual(sorted(r["false_stated"]), ["labelled_wrong", "rate", "stated", "structural_violations"])
        self.assertEqual(sorted(r["provenance"]), ["checked", "correct", "problems", "rate"])
        self.assertEqual(sorted(r["conflicts"]), ["generic", "known_detected", "known_status", "known_total"])
        self.assertEqual(sorted(r["screening"]), ["benign", "benign_wrongly_dropped", "injection_recall", "injections",
                                                   "injections_caught", "secret_recall", "secrets", "secrets_caught", "synthetic"])
        self.assertEqual(sorted(r["precision"]), ["accept", "by_stratum", "edit", "labelled", "precision_strict",
                                                   "precision_with_edits", "reject", "sampled"])
        self.assertEqual(sorted(r["exclusions"]), ["candidates", "turns"])
        self.assertEqual(r["gold_recall"]["defined_by"], E.load_gold()["defined_by"])
        self.assertEqual(r["negative_leakage"]["negative_sources"], len(E.load_gold()["negatives"]))


if __name__ == "__main__":
    unittest.main()
