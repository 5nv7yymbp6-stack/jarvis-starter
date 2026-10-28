"""The sign-in guard (backtalk/authguard.py and its three call sites): an expired login produces ONE spoken message and
cannot become a restart loop, while normal startup and transient failures behave exactly as before.

Nothing here starts Backtalk, runs run.sh, calls a model, touches credentials or loads a launch agent. The CLI is always a
fake runner; markers live in a temp directory."""
import ast
import inspect
import json
import os
import re
import stat
import subprocess
import tempfile
import unittest
from types import SimpleNamespace

from backtalk import authguard as A

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def runner(stdout="", raises=None, rc=0, seen=None):
    def run(cmd, **kw):
        if seen is not None:
            seen.append((cmd, kw))
        if raises:
            raise raises
        return SimpleNamespace(stdout=stdout, stderr="", returncode=rc)
    return run


class StatusTests(unittest.TestCase):
    def test_only_a_definite_boolean_is_an_answer(self):
        cases = ((json.dumps({"loggedIn": True, "authMethod": "claude.ai"}), A.SIGNED_IN),
                 (json.dumps({"loggedIn": False, "authMethod": "none"}), A.SIGNED_OUT),
                 (json.dumps({"loggedIn": None}), A.UNKNOWN), (json.dumps({"loggedIn": "yes"}), A.UNKNOWN),
                 (json.dumps({"loggedIn": 1}), A.UNKNOWN), (json.dumps({}), A.UNKNOWN), (json.dumps([True]), A.UNKNOWN),
                 ("", A.UNKNOWN), ("not json", A.UNKNOWN), ("{broken", A.UNKNOWN))
        for out, want in cases:
            self.assertEqual(A.signed_in_status(cli="/x/claude", run=runner(out)), want, out)

    def test_a_warning_before_the_json_is_tolerated(self):
        out = 'Warning: no stdin data received in 3s\n{"loggedIn": false}\n'
        self.assertEqual(A.signed_in_status(cli="/x/claude", run=runner(out)), A.SIGNED_OUT)

    def test_a_check_that_cannot_run_is_unknown_never_signed_out(self):
        for exc in (OSError("x"), subprocess.TimeoutExpired("c", 10), PermissionError(), RuntimeError("boom"), ValueError()):
            self.assertEqual(A.signed_in_status(cli="/x/claude", run=runner(raises=exc)), A.UNKNOWN, repr(exc))
        self.assertEqual(A.signed_in_status(cli="/x/claude", run=runner('{"loggedIn": true}', rc=1)), A.SIGNED_IN)   # the JSON decides, not rc

    def test_it_runs_the_local_status_command_with_a_timeout_and_no_stdin(self):
        seen = []
        A.signed_in_status(cli="/x/claude", run=runner('{"loggedIn": true}', seen=seen))
        (cmd, kw), = seen
        self.assertEqual(cmd, ["/x/claude", "auth", "status"])
        self.assertEqual(kw["timeout"], A.STATUS_TIMEOUT_S)
        self.assertIs(kw["stdin"], subprocess.DEVNULL)
        self.assertTrue(kw["capture_output"])

    def test_no_cli_found_is_unknown(self):
        import shutil
        from unittest import mock
        with mock.patch.object(A, "cli_path", lambda: None):
            self.assertEqual(A.signed_in_status(run=runner('{"loggedIn": false}')), A.UNKNOWN)
        with mock.patch.object(shutil, "which", lambda n: None), mock.patch.dict("sys.modules", {"claude_agent_sdk": None}):
            self.assertIsNone(A.cli_path())


class MarkerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, A.MARKER_NAME)

    def test_write_present_clear(self):
        self.assertFalse(A.marker_present(self.path))
        self.assertTrue(A.write_marker(self.path, now=1700000000))
        self.assertTrue(A.marker_present(self.path))
        self.assertEqual(read_json(self.path), {"at": 1700000000, "reason": "signed_out"})
        self.assertEqual(stat.S_IMODE(os.stat(self.path).st_mode), 0o600)
        A.clear_marker(self.path)
        self.assertFalse(A.marker_present(self.path))
        A.clear_marker(self.path)                                  # clearing nothing is fine

    def test_it_never_raises(self):
        self.assertFalse(A.write_marker(os.path.join(self.tmp.name, "no", "such", "dir", A.MARKER_NAME)))
        self.assertFalse(A.marker_present(os.path.join(self.tmp.name, "no", "such")))
        A.clear_marker(os.path.join(self.tmp.name, "no", "such"))

    def test_the_marker_holds_no_secret_and_leaves_no_temp_files(self):
        A.write_marker(self.path)
        self.assertEqual(sorted(os.listdir(self.tmp.name)), [A.MARKER_NAME])
        self.assertEqual(set(read_json(self.path)), {"at", "reason"})

    def test_the_name_is_covered_by_the_voice_glob_in_gitignore(self):
        self.assertTrue(A.MARKER_NAME.startswith(".voice_"))
        with open(os.path.join(REPO, ".gitignore"), encoding="utf-8") as f:
            self.assertIn(".voice_*", [ln.strip() for ln in f])


class StartupGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, A.MARKER_NAME)

    def gate(self, status):
        calls = []

        def fn():
            calls.append(1)
            return status
        return A.startup_gate(self.path, status_fn=fn), calls

    def test_no_marker_means_no_check_at_all(self):
        code, calls = self.gate(A.SIGNED_OUT)
        self.assertEqual((code, calls), (None, []))

    def test_marker_and_still_signed_out_stops_the_start_silently(self):
        A.write_marker(self.path)
        code, calls = self.gate(A.SIGNED_OUT)
        self.assertEqual((code, len(calls)), (A.EXIT_SIGN_IN_REQUIRED, 1))
        self.assertTrue(A.marker_present(self.path))               # stays until the login is fixed

    def test_marker_and_signed_in_clears_it_and_starts_normally(self):
        A.write_marker(self.path)
        code, _ = self.gate(A.SIGNED_IN)
        self.assertIsNone(code)
        self.assertFalse(A.marker_present(self.path))

    def test_an_unanswerable_check_never_locks_the_person_out(self):
        A.write_marker(self.path)
        code, _ = self.gate(A.UNKNOWN)
        self.assertIsNone(code)
        self.assertTrue(A.marker_present(self.path))


class DecisionTests(unittest.TestCase):
    def test_signed_out_is_authentication_speaks_once_and_exits_78(self):
        first = A.decide_connect_failure(A.SIGNED_OUT, marker_was_present=False)
        self.assertEqual((first.kind, first.exit_code, first.write_marker), ("auth", 78, True))
        self.assertEqual(first.speak, A.SIGNIN_SPOKEN)
        again = A.decide_connect_failure(A.SIGNED_OUT, marker_was_present=True)
        self.assertEqual((again.kind, again.exit_code, again.write_marker, again.speak), ("auth", 78, True, None))

    def test_everything_else_is_transient_and_unchanged(self):
        for status in (A.SIGNED_IN, A.UNKNOWN, None, "weird"):
            for marked in (False, True):
                d = A.decide_connect_failure(status, marked)
                self.assertEqual((d.kind, d.exit_code, d.write_marker, d.speak), ("transient", 1, False, A.TRANSIENT_SPOKEN),
                                 (status, marked))

    def test_the_transient_message_and_exit_code_are_the_original_ones(self):
        self.assertEqual(A.EXIT_TRANSIENT, 1)
        self.assertEqual(A.TRANSIENT_SPOKEN,
                         "Bad news. The voice and the face are fine, but I couldn't reach my brain, the Claude Code "
                         "session. Check this window for the error. The usual causes: Claude Code isn't signed in, the "
                         "internet is down, or the plan is out of usage.")
        self.assertNotEqual(A.EXIT_SIGN_IN_REQUIRED, A.EXIT_TRANSIENT)

    def test_a_decision_has_no_retry_or_delay_field(self):
        self.assertEqual({f for f in A.Decision.__dataclass_fields__}, {"kind", "speak", "exit_code", "write_marker"})


class NoRestartLoopTests(unittest.TestCase):
    """Replays main()'s two call sites with the real authguard functions, in main()'s order, and counts what a relauncher
    would see over many launches."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, A.MARKER_NAME)
        self.spoken, self.connects, self.models_loaded = [], 0, 0

    def launch(self, status, connect_ok=False):
        """One launch of main(). Returns the exit code (0 = kept running)."""
        gate = A.startup_gate(self.path, status_fn=lambda: status)
        if gate is not None:
            return gate                                           # before any model load or speech
        self.models_loaded += 1
        self.connects += 1
        if connect_ok:
            A.clear_marker(self.path)
            return 0
        marked = A.marker_present(self.path)
        d = A.decide_connect_failure(status, marked)
        if d.write_marker:
            A.write_marker(self.path)
        if d.speak:
            self.spoken.append(d.speak)
        return d.exit_code

    def test_ten_relaunches_while_signed_out_speak_once_and_load_nothing_after_the_first(self):
        codes = [self.launch(A.SIGNED_OUT) for _ in range(10)]
        self.assertEqual(codes, [78] * 10)
        self.assertEqual(self.spoken, [A.SIGNIN_SPOKEN])
        self.assertEqual((self.connects, self.models_loaded), (1, 1))

    def test_after_signing_in_the_next_launch_starts_normally_and_clears_the_marker(self):
        self.assertEqual(self.launch(A.SIGNED_OUT), 78)
        self.assertEqual(self.launch(A.SIGNED_OUT), 78)
        self.assertEqual(self.launch(A.SIGNED_IN, connect_ok=True), 0)
        self.assertFalse(A.marker_present(self.path))
        self.assertEqual(self.launch(A.SIGNED_IN, connect_ok=True), 0)

    def test_a_signed_in_failure_is_transient_each_time_and_never_writes_a_marker(self):
        codes = [self.launch(A.SIGNED_IN) for _ in range(3)]
        self.assertEqual(codes, [1, 1, 1])
        self.assertEqual(self.spoken, [A.TRANSIENT_SPOKEN] * 3)             # unchanged behavior: not suppressed
        self.assertFalse(A.marker_present(self.path))

    def test_an_unknown_status_failure_is_transient_and_does_not_mark(self):
        self.assertEqual(self.launch(A.UNKNOWN), 1)
        self.assertFalse(A.marker_present(self.path))

    def test_normal_startup_never_touches_the_status_check(self):
        calls = []
        gate = A.startup_gate(self.path, status_fn=lambda: (calls.append(1), A.SIGNED_IN)[1])
        self.assertEqual((gate, calls), (None, []))

    def test_a_marker_from_a_signed_out_run_then_a_transient_outage_stays_blocked_only_while_signed_out(self):
        self.launch(A.SIGNED_OUT)
        self.assertEqual(self.launch(A.SIGNED_IN), 1)             # cleared at the gate, then a transient connect failure
        self.assertFalse(A.marker_present(self.path))


class CallSiteTests(unittest.TestCase):
    """Static checks that main.py and watch_claude.sh use the guard in the right places and did not change anything else."""

    @classmethod
    def setUpClass(cls):
        with open(os.path.join(REPO, "backtalk", "main.py"), encoding="utf-8") as f:
            cls.main_src = f.read()
        cls.tree = ast.parse(cls.main_src)

    def fn(self, name):
        return next(n for n in ast.walk(self.tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)

    def test_main_runs_the_gate_first_and_before_claiming_the_instance_or_loading_anything(self):
        body = [ast.unparse(st) for st in self.fn("main").body]
        gate = next(i for i, t in enumerate(body) if "authguard.startup_gate" in t)
        self.assertEqual(gate, 0)
        self.assertLess(gate, next(i for i, t in enumerate(body) if "_claim_single_instance" in t))
        self.assertLess(gate, next(i for i, t in enumerate(body) if "asyncio.run(amain())" in t))
        self.assertIn("sys.exit(_gate)", "\n".join(body))
        self.assertIn("authguard.SIGNIN_PRINTED", "\n".join(body))

    def test_the_connect_failure_path_uses_the_decision_and_never_loops(self):
        src = ast.unparse(self.fn("amain"))
        i = src.index("BRAIN CONNECT")
        block = src[i - 200:i + 2200]
        for needle in ("authguard.signed_in_status", "authguard.decide_connect_failure", "SystemExit(_decision.exit_code)",
                       "authguard.write_marker", "authguard.clear_marker"):
            self.assertIn(needle, src)
        self.assertNotIn("SystemExit(1)", block)                  # the hard-coded exit is gone from the failure path
        self.assertNotIn("while ", block)
        self.assertNotIn("retry", block.lower())

    def test_the_marker_is_cleared_only_after_a_successful_warmup(self):
        src = ast.unparse(self.fn("amain"))
        self.assertLess(src.index("brain.start()"), src.index("authguard.clear_marker"))
        self.assertLess(src.index("authguard.clear_marker"), src.index("[backtalk] brain warm"))

    def test_the_original_model_selection_and_permission_wiring_are_untouched(self):
        with open(os.path.join(REPO, "backtalk", "config.py"), encoding="utf-8") as f:
            self.assertIn('"model": "claude-sonnet-5"', f.read())
        self.assertNotIn("authguard", inspect.getsource(__import__("backtalk.brain", fromlist=["x"])))
        self.assertNotIn("authguard", inspect.getsource(__import__("backtalk.router", fromlist=["x"])))

    def test_authguard_has_no_loops_sleeps_retries_or_model_calls(self):
        tree = ast.parse(inspect.getsource(A))
        self.assertEqual([n for n in ast.walk(tree) if isinstance(n, (ast.While, ast.For, ast.AsyncFor))], [])
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        self.assertFalse(names & {"sleep", "ClaudeSDKClient", "ClaudeAgentOptions", "query", "ask_stream", "retry", "keyring",
                                  "getpass", "environ", "getenv"})
        code = ast.unparse(ast.parse(inspect.getsource(A)))          # no comments or docstring: code only
        for banned in ("keychain", "security find", "getpass", "password", "oauth", "refresh"):
            self.assertNotIn(banned, code.lower())
        opens = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "open"]
        self.assertEqual(opens, [])                                   # it reads no file at all (only the CLI's JSON answer)

    def test_the_watcher_skips_the_launch_only_when_marked_and_still_signed_out(self):
        with open(os.path.join(REPO, "watch_claude.sh"), encoding="utf-8") as f:
            sh = f.read()
        self.assertIn('MARKER="$DIR/.voice_auth_required"', sh)
        self.assertIn('auth status', sh)
        self.assertIn('"loggedIn": false', sh)
        guard = sh.index('if [ -e "$MARKER" ]')
        self.assertLess(guard, sh.index('open "/Users/yousif/Desktop/jarvis/Jarvis Voice.app"'))
        self.assertIn("exit 0", sh[guard:sh.index("pgrep -x")])
        self.assertIn('pgrep -x "Claude"', sh)                    # the original trigger is intact
        self.assertEqual(subprocess.run(["bash", "-n", os.path.join(REPO, "watch_claude.sh")]).returncode, 0)

    def test_run_sh_is_unchanged_and_still_passes_the_exit_code_through(self):
        with open(os.path.join(REPO, "run.sh"), encoding="utf-8") as f:
            sh = f.read()
        self.assertIn('exec uv run python -m backtalk.main "$@"', sh)
        self.assertNotIn("while ", sh)
        self.assertNotIn("until ", sh)


if __name__ == "__main__":
    unittest.main()
