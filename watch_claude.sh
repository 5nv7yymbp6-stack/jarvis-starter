#!/bin/bash
# Watcher: if the Claude desktop app is running and the voice line is not, relaunch it through Jarvis Voice.app (the documented launcher,
# which runs run.sh inside Terminal so the microphone permission is kept). Invoked every 10 s by the com.jarvis.backtalk-watcher LaunchAgent.
#
# Run it as `/bin/bash -c '/bin/bash < /Users/yousif/Desktop/jarvis/backtalk/watch_claude.sh'`: launchd's bash is refused a script path under
# ~/Desktop by macOS privacy protection ("Operation not permitted", 26,738 times in watcher.log), but the dreaming job reads its script from
# stdin the same way and runs fine. Because the script then has no file name, DIR must not be derived from $0, and nothing here may call
# another script under ~/Desktop by path.
#
# Safety (2026-10-22 rework):
# - It NEVER kills or signals anything. It only looks, and at most opens the launcher.
# - "Running" means the voice line's own single-instance lock (TCP 8791) is held by a process whose command line is exactly
#   "$DIR/.venv/bin/python3 -m backtalk.main" (pid + identity, not a pattern match). A port held by anything else => no launch.
# - Bounded restarts: after a launch it waits GRACE_S before judging again, then backs off GRACE_S x 2^(n-1) between attempts, and after
#   MAX_LAUNCHES launches within WINDOW_S it gives up (one FAILED line) until the voice line is seen running again (a manual start).
# - Intentional stops are honoured: `touch .voice_off` (watcher off), and `.voice_quit` (written by the voice line on "goodbye Jarvis",
#   cleared by run.sh on the next launch). A closed Terminal window or a crash is NOT intentional and is restarted.
# - Log lines only on a change of state (no line every 10 s).
DIR="/Users/yousif/Desktop/jarvis/backtalk"
[ -n "$JARVIS_WATCH_DIR" ] && DIR="$JARVIS_WATCH_DIR"
PORT="${JARVIS_WATCH_PORT:-8791}"
NOW="${JARVIS_WATCH_NOW:-$(date +%s)}"
MARKER="$DIR/.voice_auth_required"
OFF="$DIR/.voice_off"
QUIT="$DIR/.voice_quit"
STAMPS="$DIR/.watcher_launches"
STATE="$DIR/.watcher_state"
MAX_LAUNCHES=3
WINDOW_S=1800
GRACE_S=90
CLI="$DIR/.venv/lib/python3.12/site-packages/claude_agent_sdk/_bundled/claude"

say() { echo "$(date '+%F %T') watcher: $*"; }
prev_state() { [ -f "$STATE" ] && head -1 "$STATE" 2>/dev/null; }
# set_state NEW "log line": write the state; print the line only when the state actually changed
set_state() {
  if [ "$(prev_state)" != "$1" ]; then
    printf '%s\n' "$1" > "$STATE" 2>/dev/null
    [ -n "$2" ] && say "$2"
  fi
}

# >>> voice identity (keep identical in run.sh)
voice_pid() {
  local p c
  for p in $(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null); do
    c=$(ps -o command= -p "$p" 2>/dev/null | sed 's/[[:space:]]*$//')
    if [ "$c" = "$DIR/.venv/bin/python3 -m backtalk.main" ]; then echo "$p"; return 0; fi
  done
  return 1
}
# <<< voice identity

[ -e "$OFF" ] && { set_state "off" "off switch present (.voice_off): not watching"; exit 0; }

# Sign-in guard: when the last run ended because the Claude Code login expired (marker present) and `claude auth status` still says
# signed out, do not relaunch. Signed in, or the check cannot answer: carry on and let the app's own startup gate decide.
if [ -e "$MARKER" ] && [ -x "$CLI" ]; then
  if "$CLI" auth status </dev/null 2>/dev/null | grep -q '"loggedIn": false'; then
    set_state "signed-out" "Claude Code is signed out: not relaunching"; exit 0
  fi
fi

PID=$(voice_pid)
if [ -n "$PID" ]; then
  [ "$(prev_state)" = "gaveup" ] && rm -f "$STAMPS"      # started by hand after a give-up: the restart budget starts fresh
  set_state "up $PID" "voice line running (pid $PID)"
  exit 0
fi
HOLDER=$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null | head -1)
if [ -n "$HOLDER" ]; then
  set_state "foreign $HOLDER" "port $PORT is held by pid $HOLDER, which is not the voice line: not launching (nothing is killed)"
  exit 0
fi

PREV=$(prev_state)
# A deliberate goodbye is checked BEFORE the DOWN line, so it is never logged as a crash (live test 2026-10-23 logged both).
if [ -e "$QUIT" ]; then
  WAS=""
  case "$PREV" in up\ *) WAS=" (was pid ${PREV#up })";; esac
  set_state "quit" "voice line stopped$WAS: hung up on purpose (.voice_quit), not relaunching until it is started by hand"
  exit 0
fi
case "$PREV" in
  up\ *) say "voice line DOWN (was pid ${PREV#up }): crash or closed window";;
esac
pgrep -x "Claude" >/dev/null 2>&1 || { set_state "no-claude" "Claude desktop app not running: not launching"; exit 0; }
[ "$PREV" = "gaveup" ] && exit 0

# Bounded restarts with backoff.
RECENT=0; LAST=0; KEEP=""
if [ -f "$STAMPS" ]; then
  while read -r T; do
    case "$T" in ''|*[!0-9]*) continue;; esac
    T=$((10#$T))
    [ "$T" -gt "$NOW" ] && continue          # a stamp from the future (clock set back) must not wedge the watcher
    if [ $((NOW - T)) -lt "$WINDOW_S" ]; then
      RECENT=$((RECENT + 1)); KEEP="$KEEP$T
"
      [ "$T" -gt "$LAST" ] && LAST=$T
    fi
  done < "$STAMPS"
fi
if [ "$RECENT" -ge "$MAX_LAUNCHES" ]; then
  set_state "gaveup" "FAILED: $RECENT launches in the last ${WINDOW_S}s and the voice line is still not running; giving up until it is started by hand (touch .voice_off to silence)"
  exit 0
fi
if [ "$RECENT" -gt 0 ]; then
  WAIT=$((GRACE_S * (1 << (RECENT - 1))))
  if [ $((NOW - LAST)) -lt "$WAIT" ]; then
    set_state "wait $RECENT" "waiting up to ${WAIT}s after launch $RECENT before trying again"
    exit 0
  fi
fi
if ! printf '%s%s\n' "$KEEP" "$NOW" > "$STAMPS" 2>/dev/null; then
  say "cannot record launches in $STAMPS; not launching (the rate limit must not fail open)"
  exit 0
fi
printf '%s\n' "launch $((RECENT + 1))" > "$STATE" 2>/dev/null
say "restart $((RECENT + 1)) of $MAX_LAUNCHES: launching Jarvis Voice"
if [ -n "$JARVIS_WATCH_OPEN" ]; then
  /bin/bash -c "$JARVIS_WATCH_OPEN"
else
  open "/Users/yousif/Desktop/jarvis/Jarvis Voice.app"
fi
