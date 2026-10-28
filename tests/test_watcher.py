"""watch_claude.sh and run.sh's replace step.

Stubbed tests: lsof, ps, pgrep and the launcher are replaced, so no real app is launched and no process is touched.
Fixture tests: a disposable fake voice line (a real python process, `<tmp>/.venv/bin/python3 -m backtalk.main`, holding a spare port) is
started, crashed and replaced; the real lsof/ps identify it. Only processes the test itself started are ever signalled."""
import os
import re
import socket
import stat
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SCRIPT = os.path.join(ROOT, "watch_claude.sh")
RUNSH = os.path.join(ROOT, "run.sh")


def read(p):
    with open(p) as f:
        return f.read()


class Stubbed(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = os.path.join(self.tmp.name, "dir")
        self.bin = os.path.join(self.tmp.name, "bin")
        os.makedirs(self.d)
        os.makedirs(self.bin)
        self.opens = os.path.join(self.tmp.name, "opens")
        self.now = 1_800_000_000

    def stub(self, name, body):
        p = os.path.join(self.bin, name)
        with open(p, "w") as f:
            f.write("#!/bin/bash\n" + body + "\n")
        os.chmod(p, os.stat(p).st_mode | stat.S_IEXEC)

    def run_watch(self, claude=True, holder=None, cmd=None, via_stdin=False, now=None):
        """holder: pid listening on the lock port (None = nobody); cmd: that pid's command line (default: the real voice line's)."""
        cmd = cmd if cmd is not None else f"{self.d}/.venv/bin/python3 -m backtalk.main"
        self.stub("pgrep", f'case "$*" in *Claude*) exit {0 if claude else 1};; esac; exit 1')
        self.stub("lsof", f'echo -n "{holder or ""}"; [ -n "{holder or ""}" ] && echo; exit 0')
        self.stub("ps", f'[ "$4" = "{holder}" ] && echo "{cmd}   "; exit 0')
        self.stub("open", f'echo "$@" >> "{self.opens}"')
        env = dict(os.environ, PATH=self.bin + ":" + os.environ["PATH"], JARVIS_WATCH_DIR=self.d,
                   JARVIS_WATCH_NOW=str(now if now is not None else self.now))
        env.pop("JARVIS_WATCH_OPEN", None)
        if via_stdin:
            with open(SCRIPT) as f:
                return subprocess.run(["/bin/bash"], stdin=f, env=env, capture_output=True, text=True)
        return subprocess.run(["/bin/bash", SCRIPT], env=env, capture_output=True, text=True)

    def n_opens(self):
        return len(read(self.opens).splitlines()) if os.path.exists(self.opens) else 0

    # normal startup / already running
    def test_launches_the_documented_app_when_claude_is_running_and_the_voice_line_is_not(self):
        r = self.run_watch()
        self.assertEqual(self.n_opens(), 1)
        self.assertEqual(read(self.opens).strip(), "/Users/yousif/Desktop/jarvis/Jarvis Voice.app")
        self.assertIn("restart 1 of 3", r.stdout)

    def test_works_when_the_script_is_read_from_stdin_as_launchd_runs_it(self):
        r = self.run_watch(via_stdin=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.n_opens(), 1)

    def test_an_identified_running_voice_line_is_left_alone(self):
        r = self.run_watch(holder=4242)
        self.assertEqual(self.n_opens(), 0)
        self.assertIn("running (pid 4242)", r.stdout)
        self.assertEqual(self.run_watch(holder=4242).stdout, "")          # no repeat line while nothing changes

    def test_a_port_held_by_something_else_is_never_launched_over(self):
        for cmd in ("/usr/bin/python3 -m http.server", "/bin/zsh -c echo backtalk.main", f"{self.d}/.venv/bin/python3 -m backtalk.main --x"):
            r = self.run_watch(holder=999, cmd=cmd)
            self.assertEqual(self.n_opens(), 0, cmd)
        self.assertIn("not the voice line", self.run_watch(holder=998, cmd="/bin/sh").stdout)

    def test_a_shell_mentioning_backtalk_main_does_not_count_as_running(self):
        # the old `pgrep -f backtalk.main` matched any such shell and so never relaunched; identity needs the lock port
        r = self.run_watch(holder=None)
        self.assertEqual(self.n_opens(), 1, r.stdout)

    def test_does_nothing_when_claude_is_not_running(self):
        self.run_watch(claude=False)
        self.assertEqual(self.n_opens(), 0)

    # crash, backoff, give-up
    def test_a_crash_is_logged_and_restarted(self):
        self.run_watch(holder=4242)
        r = self.run_watch(holder=None, now=self.now + 5)
        self.assertIn("DOWN (was pid 4242)", r.stdout)
        self.assertEqual(self.n_opens(), 1)

    def test_repeated_crashes_back_off_then_give_up_with_one_line(self):
        t = self.now
        self.run_watch(now=t)                                            # launch 1
        self.assertEqual(self.n_opens(), 1)
        self.assertIn("waiting up to 90s", self.run_watch(now=t + 10).stdout)
        self.assertEqual(self.run_watch(now=t + 20).stdout, "")          # same state: silent
        self.run_watch(now=t + 91)                                       # launch 2 after 90 s
        self.assertEqual(self.n_opens(), 2)
        self.run_watch(now=t + 91 + 179)                                 # 180 s backoff not yet over
        self.assertEqual(self.n_opens(), 2)
        self.run_watch(now=t + 91 + 181)                                 # launch 3
        self.assertEqual(self.n_opens(), 3)
        r = self.run_watch(now=t + 91 + 181 + 400)
        self.assertIn("FAILED: 3 launches", r.stdout)
        for k in range(5):
            self.assertEqual(self.run_watch(now=t + 2000 + k).stdout, "")   # given up: silent, no launches, even past the window
        self.assertEqual(self.n_opens(), 3)

    def test_a_manual_start_after_giving_up_resets_the_budget(self):
        t = self.now
        for dt in (0, 91, 91 + 181):
            self.run_watch(now=t + dt)
        self.run_watch(now=t + 1000)
        self.assertIn("gaveup", read(os.path.join(self.d, ".watcher_state")))
        self.run_watch(holder=7, now=t + 1100)                           # he started it by hand
        r = self.run_watch(now=t + 1200)                                 # it crashes again
        self.assertIn("DOWN (was pid 7)", r.stdout)
        self.assertEqual(self.n_opens(), 4)

    # intentional stops
    def test_the_off_switch_stops_the_watcher(self):
        open(os.path.join(self.d, ".voice_off"), "w").close()
        self.run_watch()
        self.assertEqual(self.n_opens(), 0)
        os.unlink(os.path.join(self.d, ".voice_off"))
        self.run_watch()
        self.assertEqual(self.n_opens(), 1)

    def test_a_goodbye_logs_only_the_intentional_stop_never_a_crash(self):
        self.run_watch(holder=4242)
        open(os.path.join(self.d, ".voice_quit"), "w").close()
        out = self.run_watch(holder=None).stdout
        self.assertEqual(len(out.splitlines()), 1, out)
        self.assertIn("voice line stopped (was pid 4242): hung up on purpose", out)
        self.assertNotIn("DOWN", out)
        self.assertNotIn("crash", out)
        self.assertEqual(self.run_watch(holder=None).stdout, "")          # and silent afterwards

    def test_a_spoken_goodbye_is_not_restarted(self):
        self.run_watch(holder=4242)
        open(os.path.join(self.d, ".voice_quit"), "w").close()
        r = self.run_watch(holder=None)
        self.assertIn("hung up on purpose", r.stdout)
        self.assertEqual(self.n_opens(), 0)

    # stamp-file edge cases (kept from the 2026-10-21 version)
    def test_old_launches_age_out_of_the_window(self):
        old = self.now - 5000
        with open(os.path.join(self.d, ".watcher_launches"), "w") as f:
            f.write(f"{old}\n{old + 1}\n{old + 2}\n")
        self.run_watch()
        self.assertEqual(self.n_opens(), 1)

    def test_junk_and_leading_zero_and_future_stamps_do_not_wedge_it(self):
        for junk in ("abc\n\n-5\n", "08\n09\n", "".join(f"{self.now + 100000 + i}\n" for i in range(3))):
            with open(os.path.join(self.d, ".watcher_launches"), "w") as f:
                f.write(junk)
            for name in (".watcher_state", self.opens):
                p = name if os.path.isabs(name) else os.path.join(self.d, name)
                if os.path.exists(p):
                    os.unlink(p)
            r = self.run_watch()
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(self.n_opens(), 1, junk)

    def test_an_unwritable_stamp_file_stops_launching_instead_of_failing_open(self):
        os.makedirs(os.path.join(self.d, ".watcher_launches"))
        r = self.run_watch()
        self.assertEqual(self.n_opens(), 0)
        self.assertIn("cannot record", r.stdout)

    # structure
    def test_the_watcher_never_signals_anything(self):
        code = "\n".join(l for l in read(SCRIPT).splitlines() if not l.lstrip().startswith("#"))
        self.assertIsNone(re.search(r"\b(kill|pkill|killall)\b", code))

    def test_run_sh_and_the_watcher_share_one_identity_check_and_no_broad_pkill_remains(self):
        block = lambda s: s[s.index("# >>> voice identity"):s.index("# <<< voice identity")].split("\n", 1)[1]
        self.assertEqual(block(read(SCRIPT)), block(read(RUNSH)))
        self.assertNotIn("pkill", "\n".join(l for l in read(RUNSH).splitlines() if not l.lstrip().startswith("#")))


FAKE_MAIN = """import os, socket, sys, time
s = socket.socket(); s.bind(("127.0.0.1", int(os.environ["FIX_PORT"]))); s.listen(1)
life = float(os.environ.get("FIX_LIFE", "600"))
time.sleep(life)
sys.exit(1)
"""


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class Fixture(unittest.TestCase):
    """Real processes, real lsof/ps; the launcher is replaced by a command that starts the disposable fake voice line."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = os.path.realpath(self.tmp.name)
        os.makedirs(os.path.join(self.d, ".venv", "bin"))
        os.symlink(os.path.realpath(sys.executable), os.path.join(self.d, ".venv", "bin", "python3"))
        os.makedirs(os.path.join(self.d, "backtalk"))
        open(os.path.join(self.d, "backtalk", "__init__.py"), "w").close()
        with open(os.path.join(self.d, "backtalk", "main.py"), "w") as f:
            f.write(FAKE_MAIN)
        self.port = free_port()
        self.bin = os.path.join(self.d, "bin")
        os.makedirs(self.bin)
        with open(os.path.join(self.bin, "pgrep"), "w") as f:                # only "is the Claude app running" is stubbed
            f.write('#!/bin/bash\ncase "$*" in *Claude*) exit 0;; esac; exec /usr/bin/pgrep "$@"\n')
        os.chmod(os.path.join(self.bin, "pgrep"), 0o755)
        self.started = []
        self.addCleanup(self.cleanup)

    def cleanup(self):
        for p in self.pids_we_started():
            try:
                os.kill(p, 9)
            except OSError:
                pass

    def pids_we_started(self):
        out = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True).stdout
        want = f"{self.d}/.venv/bin/python3 -m backtalk.main"
        return [int(l.split(None, 1)[0]) for l in out.splitlines() if l.split(None, 1)[1:] == [want]]

    def env(self, life="600", now=None):
        launch = (f'cd "{self.d}"; FIX_PORT={self.port} FIX_LIFE={life} nohup "{self.d}/.venv/bin/python3" -m backtalk.main '
                  f'</dev/null >/dev/null 2>&1 &')
        e = dict(os.environ, PATH=self.bin + ":" + os.environ["PATH"], JARVIS_WATCH_DIR=self.d, JARVIS_WATCH_PORT=str(self.port),
                 JARVIS_WATCH_OPEN=launch)
        if now is not None:
            e["JARVIS_WATCH_NOW"] = str(now)
        return e

    def watch(self, **kw):
        with open(SCRIPT) as f:                                           # stdin, as launchd runs it
            return subprocess.run(["/bin/bash"], stdin=f, env=self.env(**kw), capture_output=True, text=True, timeout=30)

    def wait_up(self, n=1, timeout=10):
        end = time.time() + timeout
        while time.time() < end:
            if len(self.pids_we_started()) == n and subprocess.run(["lsof", "-nP", f"-iTCP:{self.port}", "-sTCP:LISTEN", "-t"],
                                                                   capture_output=True, text=True).stdout.strip():
                return self.pids_we_started()
            time.sleep(0.2)
        self.fail(f"fake voice line not up: {self.pids_we_started()}")

    def test_startup_crash_restart_and_no_duplicate(self):
        t = 1_800_000_000
        self.assertIn("restart 1 of 3", self.watch(now=t).stdout)
        (pid,) = self.wait_up()
        self.assertIn(f"running (pid {pid})", self.watch(now=t + 10).stdout)
        for k in range(3):                                                # already running: never a second instance
            self.watch(now=t + 20 + k)
        self.assertEqual(self.pids_we_started(), [pid])
        os.kill(pid, 9)                                                   # simulated crash of OUR fixture process
        time.sleep(0.5)
        r = self.watch(now=t + 200)
        self.assertIn(f"DOWN (was pid {pid})", r.stdout)
        self.assertIn("restart 2 of 3", r.stdout)
        (pid2,) = self.wait_up()
        self.assertNotEqual(pid2, pid)

    def test_a_voice_line_that_keeps_dying_is_bounded(self):
        t = 1_800_000_000
        for dt in (0, 91, 91 + 181, 91 + 181 + 361, 2000, 2100):
            self.watch(life="0.3", now=t + dt)
            time.sleep(1.2)                                               # it binds, then dies
        self.assertEqual(self.pids_we_started(), [])
        stamps = read(os.path.join(self.d, ".watcher_launches")).split()
        self.assertEqual(len(stamps), 3)
        self.assertEqual(read(os.path.join(self.d, ".watcher_state")).strip(), "gaveup")

    def test_watcher_off_switch_with_a_live_fixture(self):
        open(os.path.join(self.d, ".voice_off"), "w").close()
        self.watch(now=1_800_000_000)
        time.sleep(1)
        self.assertEqual(self.pids_we_started(), [])

    def test_run_sh_replace_step_stops_only_the_identified_process(self):
        self.watch(now=1_800_000_000)
        (pid,) = self.wait_up()
        bystander = subprocess.Popen(["/bin/bash", "-c", "sleep 60 # backtalk.main backtalk[.]main"])   # what the old pkill -f would hit
        self.addCleanup(bystander.kill)
        decoy = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "-m", "backtalk.main"])
        self.addCleanup(decoy.kill)
        runsh = os.path.join(self.d, "run.sh")
        with open(RUNSH) as src, open(runsh, "w") as dst:
            dst.write(src.read())
        os.chmod(runsh, 0o755)
        open(os.path.join(self.d, ".voice_quit"), "w").close()
        r = subprocess.run(["/bin/bash", runsh], env=dict(self.env(), JARVIS_RUNSH_STOP_ONLY="1"), capture_output=True, text=True, timeout=30)
        self.assertIn(f"replaced a previous voice session (pid {pid})", r.stdout, r.stderr)
        self.assertEqual(self.pids_we_started(), [])
        self.assertIsNone(bystander.poll(), "an unrelated process was killed")
        self.assertIsNone(decoy.poll(), "an unrelated python was killed")
        self.assertFalse(os.path.exists(os.path.join(self.d, ".voice_quit")))


class QuitMarker(unittest.TestCase):
    def test_goodbye_writes_the_marker_and_never_raises(self):
        from backtalk import main as M
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, ".voice_quit")
            M.mark_intentional_quit(p)
            self.assertTrue(os.path.exists(p))
            M.mark_intentional_quit(os.path.join(d, "missing", "x"))      # unwritable: hanging up still works
        src = read(os.path.join(ROOT, "backtalk", "main.py"))
        i = src.index('mouth.say(CFG["signoff"])')
        self.assertIn("mark_intentional_quit()", src[i:i + 200])


if __name__ == "__main__":
    unittest.main()
