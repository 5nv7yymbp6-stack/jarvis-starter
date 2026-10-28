#!/bin/bash
# backstage.sh — the voice line's log, trimmed for a second session watching
# a live test (upstream's stream: show-Jarvis on stage, a backstage session
# reading along and fixing what breaks). Follows logs/backtalk.log from now
# on, keeps what was said and every line that means something happened,
# drops the per-utterance noise, and marks trouble with "!!".
#
# Run from a typed Claude session under Monitor while the person talks.
# Reads the log only; changes nothing.
cd "$(dirname "$0")"
tail -n 0 -F logs/backtalk.log 2>/dev/null |
grep --line-buffered -v -E \
  'voice_lock shadow|\[aec\] (flush|resume|pause)|\[ears\] heard [0-9.]+s|FutureWarning|UserWarning|warnings\.warn' |
grep --line-buffered -E \
  '\[(you|Jarvis|zip|console|board|brain|watchdog|perm|backtalk|turn)\]|\[ears\] (ignored|kept)|error|fail|stuck|timed out|Traceback|desync' |
sed -u -E 's/^(.*(error|Error|fail|stuck|timed out|Traceback|desync|resume failed).*)$/!! \1/'
