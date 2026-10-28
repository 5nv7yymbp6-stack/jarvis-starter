"""Memory 2.0, Phase 7: the review program (backtalk/evals/memory_extraction/review_cli.py). Scripted terminals, temp folders,
synthetic notes. Proves the rules that matter: no terminal, no decision; only the ledger is written; the vault, the
transcripts, the store, the index and the dreaming system are never touched."""
import ast
import builtins
import inspect
import json
import os
import subprocess
import sys
import unittest

from backtalk import memory_extract as X
from backtalk import memory_review as R
from backtalk import memory_store as S
from backtalk.evals.memory_extraction import review_cli as C
from backtalk.evals.memory_extraction import scan
from tests.test_memory_extract import NOW, Base, entry
from tests.test_memory_review import fixture_notes

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T = iter(range(10 ** 6))


def clock():
    n = next(T) % 3600
    return f"2026-03-01T10:{n // 60:02d}:{n % 60:02d}Z"


class Script:
    """A scripted terminal: answers in order; running out is a test failure, not a hang."""

    def __init__(self, *answers):
        self.answers, self.prompts, self.said = list(answers), [], []

    def ask(self, prompt=""):
        self.prompts.append(prompt)
        if not self.answers:
            raise AssertionError("script ran out at: " + prompt)
        return self.answers.pop(0)

    def say(self, text=""):
        self.said.append(str(text))

    @property
    def out(self):
        return "\n".join(self.said)


class CliBase(Base):
    def setUp(self):
        super().setUp()
        self.v = self.mkdir("repo", "vault")
        self.tr = self.mkdir("tr")
        self.review = os.path.join(self.root, "home", "review")
        for n in fixture_notes():
            self.write_note(self.v, n.path, n.text)
        self.write_transcript(self.tr, "aaaaaaaa-0000.jsonl", [entry("hello there my friend", uuid="t0")])
        self.make_pack(NOW)

    def make_pack(self, now):
        cands, conflicts, meta, _ = scan.run_pipeline(self.tr, self.v, now)
        return scan.write_pack(self.review, cands, conflicts, meta, forbidden=[self.v, os.path.dirname(self.v), self.tr])

    def run_cli(self, script, *extra, tty=True):
        return C.main(["--root", self.review, "--vault", self.v, "--transcripts", self.tr, *extra],
                      ask=script.ask, say=script.say, tty=tty, now_fn=clock)

    def ledger(self):
        return C.read_ledger(os.path.join(self.review, C.LEDGER))

    def actions(self):
        return [r["action"] for r in self.ledger()]


class RefusalTests(CliBase):
    def test_without_a_terminal_nothing_is_read_or_written(self):
        s = Script()
        self.assertEqual(self.run_cli(s, tty=False), 2)
        self.assertEqual(s.said, [C.NO_TTY])
        self.assertEqual(s.prompts, [])
        self.assertFalse(os.path.exists(os.path.join(self.review, C.LEDGER)))
        self.assertFalse(os.path.exists(os.path.join(self.review, C.PRIVATE)))

    def test_a_real_process_with_a_pipe_or_no_stdin_is_refused(self):
        root = self.mkdir("fresh")
        for stdin in (subprocess.DEVNULL, subprocess.PIPE):
            r = subprocess.run([sys.executable, "-m", "backtalk.evals.memory_extraction.review_cli", "--root", root, "--vault", self.v,
                                "--transcripts", self.tr], cwd=REPO, input=b"lock\nlock\n" if stdin == subprocess.PIPE else None,
                               stdin=None if stdin == subprocess.PIPE else stdin, capture_output=True, timeout=60)
            self.assertEqual(r.returncode, 2)
            self.assertIn(C.NO_TTY, r.stdout.decode())
        self.assertEqual(os.listdir(root), [])

    def test_the_command_line_has_no_way_to_claim_a_terminal(self):
        with self.assertRaises(SystemExit):
            C.main(["--root", self.review, "--tty"], tty=False)
        src = inspect.getsource(C)
        self.assertNotIn('"--tty"', src)

    def test_the_review_folder_may_not_be_inside_the_vault_the_repo_or_the_transcripts(self):
        for bad in (os.path.join(self.v, "r"), os.path.join(os.path.dirname(self.v), "r"), os.path.join(self.tr, "r")):
            with self.assertRaises(ValueError):
                C.main(["--root", bad, "--vault", self.v, "--transcripts", self.tr], ask=Script().ask, say=Script().say, tty=True)
            self.assertFalse(os.path.exists(bad))

    def test_no_pack_is_an_error(self):
        empty = self.mkdir("elsewhere")
        with self.assertRaises(R.ReviewError) as cm:
            C.main(["--root", empty, "--vault", self.v, "--transcripts", self.tr], ask=Script().ask, say=Script().say, tty=True)
        self.assertEqual(cm.exception.code, "no_pack")


class InspectTests(CliBase):
    def test_status_pilot_and_verify_need_no_terminal_and_create_nothing(self):
        before = self.snapshot(self.review)
        s = Script()
        self.assertEqual(self.run_cli(s, "--pilot", tty=False), 0)
        self.assertEqual(self.run_cli(s, "--status", tty=False), 0)
        self.assertEqual(self.run_cli(s, "--verify", tty=False), 0)
        self.assertEqual(self.snapshot(self.review), before)
        self.assertIn("6 candidates (no private, instruction or template-default candidate is ever offered)", s.out)
        self.assertIn('"offered": 6', s.out)
        self.assertIn("ledger chain intact", s.out)
        self.assertEqual(s.prompts, [])

    def test_verify_detects_a_tampered_ledger(self):
        self.run_cli(Script("quit"))
        self.run_cli(Script("quit"))
        path = os.path.join(self.review, C.LEDGER)
        text = open(path, encoding="utf-8").read().replace('"action":"session"', '"action":"sessio"')
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        s = Script()
        self.assertEqual(self.run_cli(s, "--verify", tty=False), 1)
        self.assertIn("BROKEN", s.out)
        self.assertEqual(self.run_cli(s, "--status", tty=False), 1)
        with self.assertRaises(R.ReviewError) as cm:
            self.run_cli(Script("quit"))
        self.assertEqual(cm.exception.code, "ledger_corrupt")


class SessionTests(CliBase):
    FULL = ["why", "lock", "y", "lock", "lock",                       # goal 1: why; `y` is not a decision; then really lock
            "reject", "because", "reject", "not_true",                # goal 2: bad reason refused, then rejected
            "skip",                                                   # new_ventures_priority
            "defer",                                                  # occupation
            "neither", "lock", "lock", "skip",                        # business_start: conflict resolved as neither; lock refused; skip
            "own", "edit", "Kansas City, Missouri", "", "", "n"]      # operating_focus: own; edit

    def test_a_full_session(self):
        s = Script(*self.FULL)
        self.assertEqual(self.run_cli(s), 0)
        led = self.ledger()
        self.assertEqual([r["action"] for r in led], ["open", "lock", "reject", "skip", "defer", "conflict_choice", "skip",
                                                      "conflict_choice", "edit"])
        lock, reject, skip, defer, c1, skip2, c2, edit = led[1:]
        self.assertEqual((lock["key"], lock["adoption"]["kind"], lock["origin"]["kind"]), ("goal_cash_buyer_list_deep", "adopted_note_text", "note"))
        self.assertEqual((reject["key"], reject["reason"]), ("goal_know_each_market_well", "not_true"))
        self.assertEqual((c1["choice"], c1["key"]), ("neither", "user.business_start"))
        self.assertEqual((c2["choice"], edit["value"], edit["adoption"]["kind"], edit["key"]), ("own", "Kansas City, Missouri", "typed_by_user", "operating_focus"))
        self.assertIn("refused: lock_word", s.out)
        self.assertIn("refused: bad_reason", s.out)
        self.assertIn("refused: conflict_chose_neither", s.out)
        self.assertIn("locked (recorded in the review ledger only; nothing was written to memory)", s.out)
        self.assertIn("No candidate is waiting for a first decision", s.out)
        self.assertEqual(R.verify_chain(led), [])

    def test_the_word_lock_must_be_typed_in_full_every_time(self):
        s = Script("lock", "", "lock", "LOCK", "lock", "y", "lock", "locked", "lock", "yes", "quit")
        self.run_cli(s)
        self.assertEqual(self.actions(), ["open"])
        self.assertEqual(s.out.count("refused: lock_word"), 5)
        s = Script("lock", "  lock  ", "quit")           # only surrounding whitespace is ignored (the terminal's line ending)
        self.run_cli(s)
        self.assertEqual(self.actions(), ["open", "session", "lock"])

    def test_quitting_keeps_progress_and_the_next_session_continues_without_repeating(self):
        self.run_cli(Script("lock", "lock", "quit"))
        s = Script("quit")
        self.run_cli(s)
        self.assertEqual(self.actions(), ["open", "lock", "session"])
        self.assertTrue(any("goal_know_each_market_well" in line for line in s.said))
        self.assertFalse(any("key        goal_cash_buyer_list_deep" in line for line in s.said))

    def test_only_offered_candidates_are_ever_shown(self):
        s = Script("quit")
        self.run_cli(s)
        shown = " ".join(s.said)
        for hidden in ("pref_iraqi_arabic", "rule_direct", "residence", "Test Road"):
            self.assertNotIn(hidden, shown)

    def test_a_private_edit_is_blocked_and_its_value_goes_only_to_the_private_file(self):
        s = Script("edit", "I earn $800/week from the work", "", "", "n", "quit")
        self.run_cli(s)
        rec = self.ledger()[-1]
        self.assertEqual((rec["action"], rec["commit_blocked"], rec["value"]), ("edit", "private_requires_confirmed_option", None))
        ledger_text = open(os.path.join(self.review, C.LEDGER), encoding="utf-8").read()
        self.assertNotIn("800", ledger_text)
        priv = [json.loads(l) for l in open(os.path.join(self.review, C.PRIVATE), encoding="utf-8")]
        self.assertEqual(priv, [{"seq": rec["seq"], "candidate": rec["candidate"], "value": "I earn $800/week from the work",
                                 "evidence": "I earn $800/week from the work"}])
        self.assertNotIn("800", s.out)
        self.assertEqual(os.stat(os.path.join(self.review, C.PRIVATE)).st_mode & 0o777, 0o600)

    def test_an_edit_is_re_screened(self):
        s = Script("edit", "my api key is sk-abcdefghijklmnop1234", "", "", "n", "edit", "a perfectly fine goal", "short", "", "n", "quit")
        self.run_cli(s)
        self.assertEqual(self.actions(), ["open"])
        self.assertIn("refused: screen_refused: credential_like", s.out)
        self.assertIn("refused: bad_value: evidence_unusable", s.out)

    def test_unknown_commands_decide_nothing(self):
        s = Script("approve", "y", "ok", "", "quit")
        self.run_cli(s)
        self.assertEqual(self.actions(), ["open"])
        self.assertEqual(s.out.count("not a command"), 4)

    def test_status_inside_a_session_shows_counts_only(self):
        s = Script("status", "quit")
        self.run_cli(s)
        self.assertIn('"offered": 6', s.out)

    def test_document_drift_conflicts_are_acknowledged_not_resolved(self):
        gid = next(g["id"] for g in json.load(open(os.path.join(self.review, "run-20260201000000", "review_pack.json")))["conflicts"]
                   if g["key"] == "user.fund_count")
        s = Script()
        sess = C.Session(self.review, os.path.join(self.review, "run-20260201000000"), self.v, self.tr, s.ask, s.say, clock)
        s.answers = ["nope"]
        self.assertIsNone(sess.resolve(gid))
        s.answers = ["drift"]
        rec = sess.resolve(gid)
        self.assertEqual((rec["choice"], rec["candidate"]), ("drift", None))
        self.assertIn("a difference between your ordinary notes, not a memory", " ".join(s.prompts))


class StaleAndBindingTests(CliBase):
    def test_a_source_that_changed_after_the_pack_is_recorded_stale_and_cannot_be_locked(self):
        p = os.path.join(self.v, "VAULT-INDEX.md")
        text = open(p, encoding="utf-8").read().replace("A cash buyer list deep enough in each fund's market.", "Changed after the pack was made.")
        with open(p, "w", encoding="utf-8") as f:
            f.write(text)
        s = Script("quit")
        self.run_cli(s)
        acts = self.actions()
        self.assertIn("stale", acts)
        stale = [r for r in self.ledger() if r["action"] == "stale"][0]
        self.assertEqual(stale["key"], "goal_cash_buyer_list_deep")
        self.assertIn("evidence_not_in_line", stale["source_check"])
        self.assertIn("STALE: the source no longer matches", s.out)
        self.assertNotIn("goal_cash_buyer_list_deep", [r["key"] for r in self.ledger() if r["action"] in ("lock", "edit")])

    def test_a_ledger_belongs_to_one_pack(self):
        self.run_cli(Script("quit"))
        self.make_pack("2026-02-02T00:00:00Z")
        with self.assertRaises(R.ReviewError) as cm:
            self.run_cli(Script("quit"))
        self.assertEqual(cm.exception.code, "ledger_mismatch")
        run1 = os.path.join(self.review, "run-20260201000000")
        self.run_cli(Script("quit"), "--run", run1)

    def test_a_second_writer_cannot_interleave(self):
        self.run_cli(Script("quit"))
        path = os.path.join(self.review, C.LEDGER)
        before = open(path, encoding="utf-8").read()
        with self.assertRaises(R.ReviewError) as cm:
            C.append_line(path, {"action": "x"}, expect_prev="f" * 64)
        self.assertEqual(cm.exception.code, "ledger_changed")
        self.assertEqual(open(path, encoding="utf-8").read(), before)
        last = self.ledger()[-1]["hash"]
        C.append_line(path, {"action": "x"}, expect_prev=last)
        self.assertEqual(len(self.ledger()), 2)

    def test_an_unreadable_ledger_line_is_corruption(self):
        self.run_cli(Script("quit"))
        with open(os.path.join(self.review, C.LEDGER), "a", encoding="utf-8") as f:
            f.write("{not json\n")
        with self.assertRaises(R.ReviewError) as cm:
            C.read_ledger(os.path.join(self.review, C.LEDGER))
        self.assertEqual(cm.exception.code, "ledger_corrupt")
        self.assertEqual(C.read_ledger(os.path.join(self.review, "nothing.jsonl")), [])

    def test_a_transcript_candidate_goes_stale_when_the_turn_changes(self):
        self.write_transcript(self.tr, "bbbbbbbb-0000.jsonl", [entry("I prefer short answers when we talk.", uuid="t1")])
        done = self.make_pack("2026-02-03T00:00:00Z")
        sess = C.Session(self.review, done["dir"], self.v, self.tr, Script().ask, Script().say, clock)
        cid = next(c for c, v in sess.pack.views.items() if v["origin"] == "transcript")
        self.assertEqual(sess.problems(cid), [])
        self.write_transcript(self.tr, "bbbbbbbb-0000.jsonl", [entry("I prefer long answers when we talk.", uuid="t1")])
        self.assertIn("evidence_not_in_turn", sess.problems(cid))


class NoPersistentWriteTests(CliBase):
    def test_a_full_session_writes_only_the_ledger_files(self):
        before = (self.snapshot(self.v), self.snapshot(self.tr))
        pack_files = self.snapshot(os.path.join(self.review, "run-20260201000000"))
        self.run_cli(Script(*SessionTests.FULL))
        self.assertEqual((self.snapshot(self.v), self.snapshot(self.tr)), before)
        self.assertEqual(self.snapshot(os.path.join(self.review, "run-20260201000000")), pack_files)
        self.assertEqual(sorted(n for n in os.listdir(self.review) if not n.startswith("run-")), [C.LEDGER])
        self.assertFalse(os.path.exists(os.path.join(self.v, "11 - Memory")))
        self.assertFalse(os.path.exists(os.path.join(os.path.dirname(self.v), "11 - Memory")))

    def test_permissions(self):
        self.run_cli(Script("lock", "lock", "quit"))
        self.assertEqual(os.stat(self.review).st_mode & 0o777, 0o700)
        self.assertEqual(os.stat(os.path.join(self.review, C.LEDGER)).st_mode & 0o777, 0o600)

    def test_the_review_never_opens_the_private_companion_or_anything_in_the_vault_for_writing(self):
        opened = []
        real = builtins.open

        def spy(file, mode="r", *a, **k):
            opened.append((str(file), mode))
            return real(file, mode, *a, **k)
        builtins.open = spy
        try:
            self.run_cli(Script(*SessionTests.FULL))
        finally:
            builtins.open = real
        self.assertFalse([p for p, m in opened if "private_review" in p])
        self.assertFalse([p for p, m in opened if any(c in m for c in "wax+")])
        self.assertTrue(os.path.exists(os.path.join(self.review, "run-20260201000000", "private_review.json")))

    def test_the_program_cannot_reach_a_store_the_network_or_a_model(self):
        for mod in (C, R):
            tree = ast.parse(inspect.getsource(mod))
            imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
            imported |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
            self.assertFalse(imported & {"memory_store", "requests", "socket", "subprocess", "urllib", "httpx", "ollama"}, mod.__name__)
            called = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
            self.assertFalse(called & {"put", "apply"}, mod.__name__)
            self.assertNotIn("memory_store", {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}, mod.__name__)

    def test_the_only_writer_in_the_program_is_the_ledger_appender(self):
        tree = ast.parse(inspect.getsource(C))
        for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
            writes = []
            for n in ast.walk(fn):
                if isinstance(n, ast.Call):
                    f = n.func
                    owner = getattr(getattr(f, "value", None), "id", "")
                    name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
                    if owner == "os" and name in ("write", "makedirs", "fchmod", "replace", "unlink", "mkdir", "chmod", "remove", "rename"):
                        writes.append(name)
                    if name == "open" and owner != "os":
                        mode = next((a.value for a in n.args[1:2] if isinstance(a, ast.Constant)), "r")
                        if any(c in mode for c in "wax+"):
                            writes.append("open-" + mode)
            if writes:
                self.assertIn(fn.name, ("append_line", "main"), (fn.name, writes))

    def test_the_real_store_and_vault_are_untouched_by_the_tests(self):
        # The real folder may legitimately exist since the 2026-10-19 pilot; the test only proves the CLI run changes nothing.
        real = os.path.join(S.REAL_VAULT, S.MEMORY_DIR)

        def state():
            if not os.path.exists(real):
                return None
            return sorted((n, os.stat(os.path.join(real, n)).st_mtime_ns) for n in os.listdir(real))

        before = state()
        self.run_cli(Script(*SessionTests.FULL))
        self.assertEqual(state(), before)


class FakeStream:
    def __init__(self, tty):
        self.tty = tty

    def isatty(self):
        return self.tty


class ResidualCliTests(CliBase):
    def session(self, *answers):
        s = Script(*answers)
        return C.Session(self.review, os.path.join(self.review, "run-20260201000000"), self.v, self.tr, s.ask, s.say, clock), s

    def group(self, key):
        pk = json.load(open(os.path.join(self.review, "run-20260201000000", "review_pack.json"), encoding="utf-8"))
        return next(g["id"] for g in pk["conflicts"] if g["key"] == key)

    def test_the_exact_prompts_and_messages(self):
        s = Script("why", "lock", "lock", "edit", "I am working Kansas City these days", "", "", "n", "reject", "outdated", "skip", "back", "quit")
        self.run_cli(s)
        P = "[lock | edit | reject | skip | defer | why | status | quit] > "
        self.assertEqual(s.prompts[:3], [P, P, "Type the word  lock  to adopt this exactly as shown: "])
        self.assertEqual(s.prompts[4:8], ["your value (in your words): ", "the sentence it rests on (blank = same as the value): ",
                                          "key (blank = goal_know_each_market_well): ", "is it private? [n/y] "])
        self.assertEqual(s.prompts[9], "why? not_true / outdated / not_mine / too_private / wrong_type > ")
        self.assertIn("locked (recorded in the review ledger only; nothing was written to memory)", s.said)
        self.assertIn("edited (recorded in the review ledger only; nothing was written to memory)", s.said)
        self.assertEqual(s.said.count("=" * 78) >= 3, True)
        self.assertEqual(s.said.count("-" * 78) >= 3, True)
        self.assertEqual(C.NO_TTY, "A review decision needs a real terminal for both input and output. Nothing was read or written.")
        self.assertEqual((C.LEDGER, C.PRIVATE), ("ledger.jsonl", "ledger_private.jsonl"))

    def fresh(self):
        for name in (C.LEDGER, C.PRIVATE):
            if os.path.exists(os.path.join(self.review, name)):
                os.unlink(os.path.join(self.review, name))

    def test_choosing_a_side_by_number(self):
        gid = self.group("user.operating_focus")
        sess, s = self.session("1")
        rec = sess.resolve(gid)
        self.assertEqual((rec["choice"], rec["label"], rec["candidate"] is not None), ("side", "columbus_chosen", True))
        shown = "\n".join(s.said)
        self.assertIn("  1. columbus_chosen: 04 - Wholesaling/04 - Wholesaling.md line 1 - \"Chosen market: **Columbus, Ohio**\"", shown)
        self.assertIn("  2. columbus_open_task: Active Priorities.md line 1 (2026-09-14) - \"Wholesaling restart: pull probate leads in Columbus\"", shown)
        self.assertIn("  4. kansas_city_channel: 03 - Markets/Kansas City (Missouri).md line 2 - ", shown)
        self.assertIn("CONFLICT " + gid + "  user.operating_focus  [memory]  (nothing is decided for you)", shown)

    def test_the_last_side_the_edges_and_the_keywords(self):
        gid = self.group("user.operating_focus")
        sess, _ = self.session("4")
        self.assertEqual(sess.resolve(gid)["choice"], "own")
        for bad in ("0", "5", "-1", "abc", "", "1.5"):
            self.fresh()
            sess, _ = self.session(bad)
            self.assertIsNone(sess.resolve(gid), bad)
        for word in ("own", "both", "neither"):
            self.fresh()
            sess, _ = self.session(word)
            self.assertEqual(sess.resolve(gid)["choice"], word)

    def test_a_missing_quote_shows_private(self):
        pk = json.load(open(os.path.join(self.review, "run-20260201000000", "review_pack.json"), encoding="utf-8"))
        gid = next(g["id"] for g in pk["conflicts"] if g["kind"] == "generic")
        sess, s = self.session("back")
        sess.resolve(gid)
        self.assertIn('- "[private]"', "\n".join(s.said))

    def test_why_shows_the_item_again_and_a_yes_to_private_raises_sensitivity(self):
        s = Script("why", "edit", "a perfectly harmless goal", "", "", "y", "quit")
        self.run_cli(s)
        self.assertEqual(sum(1 for l in s.said if l.startswith("key        user.goal_cash_buyer_list_deep")), 2)
        rec = self.ledger()[-1]
        self.assertEqual((rec["action"], rec["sensitivity"], rec["commit_blocked"], rec["value"]), ("edit", "private", "private_requires_confirmed_option", None))
        priv = [json.loads(l) for l in open(os.path.join(self.review, C.PRIVATE), encoding="utf-8")]
        self.assertEqual(priv[0]["value"], "a perfectly harmless goal")

    def test_the_review_loop_continues_after_a_stale_item(self):
        p = os.path.join(self.v, "VAULT-INDEX.md")
        text = open(p, encoding="utf-8").read().replace("A cash buyer list deep enough in each fund's market.", "Changed after the pack.")
        with open(p, "w", encoding="utf-8") as f:
            f.write(text)
        s = Script("quit")
        self.run_cli(s)
        self.assertEqual(s.prompts, ["[lock | edit | reject | skip | defer | why | status | quit] > "])
        stale_line = next(l for l in s.said if l.startswith("STALE: the source no longer matches ("))
        self.assertTrue(stale_line.endswith("). Re-extract before it can be reviewed."))
        self.assertRegex(stale_line, r"\(evidence_not_in_line(, \w+)*\)")
        self.assertEqual(self.actions(), ["open", "stale"])

    def test_status_inside_a_session_is_the_exact_summary(self):
        s = Script("status", "quit")
        self.run_cli(s)
        raw = json.load(open(os.path.join(self.review, "run-20260201000000", "review_pack.json"), encoding="utf-8"))
        state = R.State(self.ledger())
        want = json.dumps(R.summary(R.Pack(raw), state), sort_keys=True)
        self.assertIn(want, s.said)

    def test_the_terminal_check_needs_both_input_and_output(self):
        real_in, real_out = sys.stdin, sys.stdout
        results = {}
        try:
            for a in (True, False):
                for b in (True, False):
                    sys.stdin, sys.stdout = FakeStream(a), FakeStream(b)
                    s = Script("quit")
                    code = C.main(["--root", self.review, "--vault", self.v, "--transcripts", self.tr], ask=s.ask, say=s.say, now_fn=clock)
                    results[(a, b)] = code
        finally:
            sys.stdin, sys.stdout = real_in, real_out
        self.assertEqual(results, {(True, True): 0, (True, False): 2, (False, True): 2, (False, False): 2})

    def test_status_and_pilot_print_their_own_formats(self):
        a = Script()
        self.run_cli(a, "--pilot", tty=False)
        self.assertEqual(len(a.said), 7)
        self.assertTrue(a.said[0].startswith("cand_") and "[pending]" in a.said[0])
        self.assertNotIn('"offered"', a.out)
        b = Script()
        self.run_cli(b, "--status", tty=False)
        raw = json.load(open(os.path.join(self.review, "run-20260201000000", "review_pack.json"), encoding="utf-8"))
        self.assertEqual(json.loads(b.said[0])["offered"], 6)
        self.assertTrue(b.said[0].startswith("{\n  \"by_review_class\""))
        self.assertEqual(len(b.said), 1)
        c = Script()
        self.run_cli(c, "--verify", tty=False)
        self.assertEqual(c.said, ["ledger chain intact"])

    def test_broken_ledger_messages(self):
        self.run_cli(Script("quit"))
        self.run_cli(Script("quit"))
        path = os.path.join(self.review, C.LEDGER)
        recs = C.read_ledger(path)
        recs[1]["run"] = "tampered"
        with open(path, "w", encoding="utf-8") as f:
            f.write("".join(R.canonical_json(r) + "\n" for r in recs))
        v = Script()
        self.assertEqual(self.run_cli(v, "--verify", tty=False), 1)
        self.assertEqual(v.said, ["ledger chain BROKEN: hash:1"])
        w = Script()
        self.assertEqual(self.run_cli(w, "--pilot", tty=False), 1)
        self.assertEqual(w.said, ["ledger chain BROKEN: hash:1"])

    def test_the_latest_run_ignores_folders_that_are_not_packs(self):
        os.makedirs(os.path.join(self.review, "run-zzz-no-pack"))
        os.makedirs(os.path.join(self.review, "other"))
        with open(os.path.join(self.review, "other", "review_pack.json"), "w", encoding="utf-8") as f:
            f.write("{}")
        self.assertTrue(C.latest_run(self.review).endswith("run-20260201000000"))

    def test_a_review_root_that_does_not_exist_is_created_private(self):
        fresh = os.path.join(self.root, "fresh", "root")
        s = Script("quit")
        C.main(["--root", fresh, "--run", os.path.join(self.review, "run-20260201000000"), "--vault", self.v, "--transcripts", self.tr],
               ask=s.ask, say=s.say, tty=True, now_fn=clock)
        self.assertEqual(os.stat(fresh).st_mode & 0o777, 0o700)
        self.assertTrue(os.path.exists(os.path.join(fresh, C.LEDGER)))

    def test_the_default_clock_is_a_real_timestamp(self):
        s = Script("quit")
        C.main(["--root", self.review, "--vault", self.v, "--transcripts", self.tr], ask=s.ask, say=s.say, tty=True)
        self.assertRegex(self.ledger()[0]["at"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

    def test_ledger_details_and_unicode_values(self):
        self.run_cli(Script("edit", "caf\u00e9 au lait", "", "", "n", "quit"))
        rec = self.ledger()[-1]
        self.assertEqual((rec["action"], rec["value"]), ("edit", "caf\u00e9 au lait"))
        raw = open(os.path.join(self.review, C.LEDGER), encoding="utf-8").read()
        self.assertIn("caf\\u00e9", raw)
        open(os.path.join(self.review, C.LEDGER), "a", encoding="utf-8").write("{bad\n")
        with self.assertRaises(R.ReviewError) as cm:
            C.read_ledger(os.path.join(self.review, C.LEDGER))
        self.assertEqual(cm.exception.details, ("unreadable_line",))

    def test_the_pack_hash_is_of_the_text_read(self):
        raw, pack_sha, run = C.read_pack(os.path.join(self.review, "run-20260201000000"))
        text = open(os.path.join(self.review, "run-20260201000000", "review_pack.json"), encoding="utf-8").read()
        self.assertEqual((pack_sha, run, raw["format"]), (X.sha(text), "run-20260201000000", X.PACK_FORMAT))


if __name__ == "__main__":
    unittest.main()
