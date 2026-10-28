# backtalk: talk to your Claude Code agent out loud.
# Copyright (C) 2026 Jared Rhodenizer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Configuration — backtalk.json in the repo root, merged over defaults.

backtalk deliberately owns NO personality. Your agent's identity lives in
the CLAUDE.md of whatever folder `agent_dir` points at — backtalk just
gives that agent a mouth and ears. The only voice-related instruction it
adds is the spoken-delivery discipline below, which is about the MEDIUM
(writing for the ear), never the character.
"""
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# One install, more than one assistant. Point BACKTALK_CONFIG at a different
# JSON file and you get a second agent (its own name, voice, folder and
# greeting) without a second copy of the code. A launcher exports it; nothing
# else changes.
CONFIG_PATH = Path(os.environ.get("BACKTALK_CONFIG") or (REPO / "backtalk.json"))

DEFAULTS = {
    # The folder whose CLAUDE.md defines WHO your agent is. The voice
    # session runs there, so it's the same assistant as your terminal
    # sessions — same name, same personality, same memory.
    "agent_dir": "~",
    # Display name, used in logs and to build the quit phrases
    # ("goodbye <name>" hangs up). Match your agent's actual name.
    "name": "Assistant",
    # The brain. Full model id ON PURPOSE — never a bare alias like
    # "sonnet": the SDK resolves aliases through its own bundled CLI and
    # can silently land on an older model. The fast tier is most of the
    # speed difference people ask about; a deep-work model makes every
    # reply noticeably slower and burns usage doing it.
    "model": "claude-sonnet-5",
    # The deep-work model for the voice console's "switch to the deep
    # model" command ("back to the fast model" returns to "model"
    # above). Full id ON PURPOSE, same reasoning as "model". The switch
    # lasts one session and is always spoken; this default never moves
    # by itself.
    "deep_model": "claude-opus-5",
    # Which of your agent's skills the voice session can SEE. null keeps the
    # CLI's own default (all of them). [] hides every one. A list names the
    # ones to allow.
    #
    # This matters on a shared screen. Skill DESCRIPTIONS live in the system
    # prompt, so if yours name clients, employers or systems, they are one
    # screen-share away from an audience. A context filter, not a sandbox:
    # it decides what the session is TOLD about, not what it can reach.
    "visible_skills": None,
    # Extra folders the agent may access beyond agent_dir (e.g. your
    # notes vault). Absolute paths or ~ paths.
    "extra_dirs": [],
    # Hold-to-talk key. Named keys ("home", "f13", "right_alt", ...)
    # or a single character.
    "ptt_key": "home",
    # The microphone mode. "ptt" (push to talk, the default and the
    # recommendation): the mic is closed except while the key is held,
    # so room audio and your own speakers can never trigger the agent.
    # "open" (hands-free listening): always listening with voice
    # detection; a video, music with vocals, or another person in the
    # room CAN trigger it, and with open speakers it can hear itself
    # (headphones recommended). The key still works in hands-free
    # listening: it interrupts, and holding it always gets you heard.
    # Switch live by voice: "go hands free" / "push to talk mode"
    # (the switch saves itself here). The --open-mic launch flag
    # forces "open" for one session.
    "mic_mode": "ptt",
    # Playback speed for the built-in voice: 1.0 is Kokoro's native
    # pace, 1.15 is noticeably brisker, 0.9 is slower. Kokoro's own
    # pipeline implements it, so quality holds across sane values
    # (roughly 0.7 to 1.5). ElevenLabs pace lives in the master chain's
    # atempo instead. (Grew out of a community proposal, issue #1.)
    "speed": 1.0,
    # Which processor runs the built-in voice (Kokoro): "cpu" (default) or
    # "mps" / "auto" for Apple Silicon's GPU. CPU gets the first word out
    # sooner: the GPU recompiles for every new sentence length, which costs
    # more than the CPU's whole synthesis, and it leaves the GPU to Whisper.
    # See mouth._pick_device for the measurements.
    "tts_device": "cpu",
    # Resume the previous conversation on launch. OFF by default: a
    # fresh session every launch is the predictable behavior. Set true
    # and backtalk saves the session id after every completed turn
    # (signals_dir/.backtalk_session) and reattaches to it at the next
    # launch, so killing the window stops costing you the conversation.
    # A resume that fails falls back to a fresh session and says so in
    # the log. (Grew out of the same community proposal, issue #1.)
    "resume_last_session": False,
    # Publish your Claude usage (the five-hour and weekly windows) on the
    # signal bus so a face can draw it. OFF by default and deliberately
    # so: this is your own account spend, and the faces this feeds are
    # frequently on a stream or a shared screen. Nothing is collected at
    # all while this is false. (Community fix, ai-visualizer issue #1.)
    "show_usage": False,
    # Reasoning effort for the voice session: "" inherits the model's
    # default; "low" / "medium" / "high" / "max" applies at launch.
    # Saying "set effort to X" in a voice session saves itself here.
    "effort": "",
    # The voice (Kokoro, local, free). bm_lewis is the proven default —
    # British male, the butler register. Others: bm_george, bm_daniel,
    # bm_fable, am_michael, af_heart... The first letter picks the
    # language pipeline (a=American, b=British, e/f/h/i/j/p/z = other
    # languages), so keep voice and accent matched.
    "voice": "bm_lewis",
    # Speech recognition (faster-whisper, local, free).
    # Models: tiny.en / base.en / small.en / medium.en — bumped from
    # small.en to medium.en on 2026-09-18 after a night of garbled/dropped
    # short utterances; small.en is the accuracy/speed sweet spot on a
    # normal machine but medium.en is markedly more reliable on short or
    # ambient-noise clips, at some extra latency/CPU cost.
    "stt_model": "medium.en",
    # "auto" uses CUDA when present, otherwise CPU. int8 keeps CPU fast.
    "stt_device": "auto",
    "stt_compute": "int8",
    # The microphone to record from, matched by NAME. "" means whatever
    # the OS calls the default input, which is right on most machines.
    #
    # Set a real device name to PIN the mic, so a headset connecting for
    # OUTPUT cannot steal your input -- which also keeps a Bluetooth
    # headset in high-quality A2DP instead of dropping it to the
    # narrowband call profile mid-sentence, degrading what you hear at
    # the same moment it takes your voice.
    #
    # A name and never an index: indices shift every time a device
    # connects or disconnects, the exact event this setting exists to
    # survive. Exact name wins, then the first case-insensitive
    # substring. A name matching nothing falls back to the default and
    # logs the inputs it did find; the mic degrades, it never goes mute.
    #
    # NOT "stt_device" below, which is the Whisper COMPUTE device.
    "mic_device": "",
    # Optional premium voice: ElevenLabs on YOUR key. The key NEVER
    # goes in a file: it's read from the macOS Keychain (item
    # `backtalk-elevenlabs`) or Linux secret-tool, with the
    # ELEVENLABS_API_KEY env var as last-resort fallback — see
    # mouth._get_elevenlabs_key for the seeding one-liners. Kokoro
    # remains the automatic fallback, so the voice degrades instead of
    # going mute if the cloud fails. Needs ffmpeg on the PATH.
    "elevenlabs": {
        "enabled": False,
        "voice_id": "",
        # Purely for you. Voice IDs are unreadable six months later, so put
        # the human name here; nothing reads it.
        "voice_note": "",
        "model": "eleven_turbo_v2_5",
        # Which OS credential-store entry holds the key. Change it if you
        # already keep an ElevenLabs key under a name of your own rather
        # than seeding a second copy of the same secret.
        "key_slot": "backtalk-elevenlabs",
        # Local mastering: ElevenLabs' site previews are mastered demo
        # clips and the raw API never matches them. This chain closes
        # the gap: presence lift, light chest, broadcast compression,
        # limiter. atempo is the one pace dial (1.0 = native).
        "master": ("atempo=1.12,highpass=f=70,"
                   "equalizer=f=3200:t=q:w=1.2:g=3.5,"
                   "equalizer=f=140:t=q:w=1:g=1.5,"
                   "acompressor=threshold=-18dB:ratio=2.5:attack=8:"
                   "release=120:makeup=4dB,alimiter=limit=0.95"),
    },
    # Speaker verification: when enabled, an utterance that doesn't
    # match the enrolled voiceprint is dropped silently, exactly like
    # nobody spoke -- it never reaches transcription or the agent.
    # Enroll (or re-enroll) with `uv run python -m backtalk.voiceid`,
    # which records a few clips and saves voiceprint.npy in the repo
    # root. OFF by default and fails OPEN: no profile enrolled, or the
    # embedding model fails to load, and every utterance passes through
    # rather than the voice line going silently deaf. threshold is a
    # cosine-similarity cutoff (0-1); 0.75 is the Resemblyzer-recommended
    # same-speaker threshold -- raise it to be stricter, lower it if your
    # own voice keeps getting rejected.
    "voice_lock": {
        "enabled": True,
        "threshold": 0.75,
    },
    # Where the signal-bus files are written (.voice_state,
    # .voice_waveform, .voice_loading_pid) — anything can watch them;
    # visualizers pair with this contract. Default: the repo root.
    "signals_dir": "",
    # THE BAREHANDS SEAM: point this at a barehands checkout's state/
    # folder and its on-screen ring becomes your agent's face — it
    # breathes while idle, spins while thinking, pulses with the voice.
    # (github.com/jaredrhod/barehands)
    "barehands_state_dir": "",
    # The HOLO board's snapshot URL (barehands server.py /api/state). Each
    # turn, what changed on the board since the last turn rides along with
    # what was said (backtalk/board.py). Empty = off.
    "board_url": "",
    # Sound played while the agent thinks, so a long pause never reads as
    # a dead line. The bundled one ships in assets/; a relative path
    # resolves against this repo. Set "" to think in silence.
    "thinking_sound": "assets/thinking.wav",
    # Spoken lines. {name} is replaced with "name" above.
    "greeting": "Voice line online. Hold {ptt_key} and talk to me.",
    # Spoken instead of "greeting" when mic_mode is "open", where telling
    # someone to hold a key is wrong. Leave "" to use "greeting" for both.
    "greeting_open_mic": "",
    "signoff": "Voice line closing. I'll be here when you need me.",
    # Appended to the spoken-delivery discipline below. The discipline covers
    # the MEDIUM (write for the ear, no markdown, keep it short); your agent's
    # CLAUDE.md covers the character. Use this for a note that belongs to
    # neither, e.g. a rule that only applies when it is speaking.
    "discipline_append": "",
    # How long a pause ends your turn in hands-free listening, in ms. Upstream
    # was 480, which cut Yousif off at every breath mid-sentence and the agent
    # answered the fragment (2026-09-14); 1000 lets a normal pause pass. Bumped
    # to 1600 (2026-09-16) so a longer mid-sentence breath while talking to
    # someone else (not the agent) doesn't get read as a finished turn either.
    # Dropped to 600 (2026-09-23) to cut perceived reply latency, on Yousif's
    # explicit call after being shown this history -- watch for the exact
    # 2026-09-14 symptom (cut off mid-sentence, agent answers a fragment)
    # coming back; if it does, this is the first value to raise again.
    "silence_ms": 600,
    # Echo cancellation through Apple voice processing (aec/jarvis-aec, see
    # backtalk/aec.py). The mic hears the person and not the voice coming
    # back out of the speakers, so the voice stops the moment the person
    # starts talking. "mic" and "speaker" are device NAMES. With
    # only_on_speakers, it stays off whenever the default output isn't that
    # speaker (headphones have no echo to cancel).
    "echo_cancel": {
        "enabled": False,
        "mic": "MacBook Air Microphone",
        "speaker": "MacBook Air Speakers",
        "only_on_speakers": True,
    },
    # Ignore what the Mac itself is playing (aec/jarvis-systap, see
    # backtalk/systap.py): a line whose words match what the Mac played at
    # that moment (a video, music) is dropped instead of answered.
    "ignore_mac_audio": {
        "enabled": False,
    },
    # Don't reattach to a saved conversation bigger than this many tokens
    # (see sessions.py). Resuming is billed on every turn and re-cached at
    # 1.25x after an idle gap, so a session left to grow quietly becomes the
    # single biggest line in the bill. 0 switches the cap off.
    "resume_max_tokens": 150000,
    # SESSION BRIEF (brief.py): a small block of stable vault context added
    # to the system prompt once per client start. Context only.
    "session_brief": {"enabled": True},
    # ADDRESSEE GATE (addressee.py), open mic only. "wake": speech reaches the
    # agent only after the wake phrase (the name must open the sentence), then
    # follow-ups need none until active_window_s of no human follow-up passes
    # or max_active_s since the last wake. The agent speaking never extends it. "always": no gate (development/testing;
    # also the --no-wake launch flag). wake_words empty = the agent's name and
    # "hey <name>". heard_as = spellings Whisper produces for the name. This
    # is an addressee gate, NOT speaker authentication.
    "addressee": {
        "mode": "wake",
        "wake_words": [],
        "heard_as": ["jarvus", "jervis", "javanis"],
        "active_window_s": 180,   # idle window: extended by human follow-ups only
        "max_active_s": 300,      # hard cap from the last explicit address
    },
    # BACKGROUND WORK + THE ACTION LEDGER (router.py, ledger.py). Background
    # agent tasks do NOT survive a session rebuild, a clean disconnect or an
    # interrupt(): interrupt() kills every background task. protect_on_barge_in
    # therefore keeps a barge-in from calling interrupt() while a background
    # task is live: the current answer is muted and allowed to finish, and
    # barge_escalate_s later the interrupt is forced anyway (killing the
    # tasks, recorded as KILLED_BY_INTERRUPT). echo_timeout_s bounds how long
    # a sent query may wait for its run to start before it is failed (never
    # completed by "the next result"). ledger.dir "" = <repo>/ledger.
    "background": {
        "protect_on_barge_in": True,
        "barge_escalate_s": 20,
        "echo_timeout_s": 20,
    },
    "ledger": {"dir": ""},
    # Proactive behavior (roadmap Phase F): reminders, scheduled checks, spoken background completions. DEFAULT OFF.
    # Only the JSON value true for "enabled" turns it on. See backtalk/proactive.py for every rule.
    # Conversation quality (roadmap Phase D). DEFAULT OFF. Only the JSON value true turns an item on.
    "conversation": {"backchannel": False},
    # Desktop presence (roadmap Phase G): enum-only readings (idle, front-app category, lock/display, output class, fresh hand count). DEFAULT OFF.
    "presence": {"enabled": False},
    "gestures": {"hush": False},
    "proactive": {"enabled": False, "urgent": "off", "act_allow": [], "quiet_hours": None, "tz": "America/Chicago"},
}

# The spoken-delivery discipline — the MEDIUM half of what used to be a
# persona. The CHARACTER half deliberately is not here: it's whatever
# lives in the agent_dir's CLAUDE.md. One identity, one place.
DISCIPLINE = (
    "VOICE SESSION (your reply is spoken aloud through a TTS engine, "
    "not displayed): you are SPEAKING, in your own voice and "
    "personality — your CLAUDE.md is who you are. The TTS engine "
    "PERFORMS your punctuation, so write like a performance, never "
    "like a memo: contractions always, punchy conversational "
    "sentences, and if a line could open a quarterly report, rewrite "
    "it like you're telling a friend. Keep it spoken, not written. "
    "DON'T ANNOUNCE THAT YOU'RE ABOUT TO ANSWER. You are talking to a "
    "friend, and a friend never says 'On it.' or 'Let me look.' or "
    "'Checking now.' before every single reply — that is a call-centre "
    "script, and it is the most robotic thing you can possibly do. An "
    "earlier version of this prompt ordered exactly that and handed you "
    "those three phrases; Yousif heard it on every turn and hated it, "
    "and the measurements showed it bought no speed at all. So: if the "
    "answer is seconds away, just answer. Say nothing first. If the "
    "work genuinely runs long, you may say something real and specific "
    "in your own words the way a person would when they get stuck — but "
    "only when it actually earns saying, never the same words twice, "
    "and never a line that carries no information. Silence for a beat "
    "is more human than a stock phrase. "
    "GARBLED NAME OR TITLE? GUESS IT, DON'T INTERROGATE HIM. The mic "
    "mangles names constantly, and when it does, say the single most "
    "likely real thing it could be and act on it — 'Iron Man 3?' and "
    "then search — instead of handing the problem back. An open 'is "
    "that the actual name, or should I take a different guess' is "
    "useless to him: he already said the words once, and repeating them "
    "won't make the mic hear better. Asking twice in a row is worse; it "
    "happened on 2026-09-22 with 'Iromani 3' and it read as stupid, not "
    "careful. Say your best guess out loud so he can correct one word, "
    "and only stop dead if the stakes are real (a command that spends "
    "money, deletes something, or contacts a person). "
    "NEVER RESTATE THE QUESTION. He just said it; saying it back to "
    "him burns the one thing he is actually waiting for. No 'So you're "
    "asking whether', no 'Good question', no summary of what he wants "
    "before you give it. First word out is the answer. "
    "ASK ONCE, THEN STOP. One question a turn, and it is the last "
    "thing you say. Never ask the same thing twice, never re-confirm "
    "what he just confirmed, and never ask permission for something he "
    "already told you to do. A clear instruction, code edits included, "
    "means do it now; ask first only before deleting data, spending "
    "money, contacting a person, or deploying to a server. "
    "TURN-TAKING, like a person on a call: default to one "
    "or two short sentences, about twenty-five words. Give ONE answer "
    "or ONE step, then stop and let them talk; never stack steps, "
    "options or follow-ups. Go longer only when they ask for detail or "
    "a walkthrough. If what you heard sounds cut off mid-thought, say "
    "only 'Go on.' and wait. "
    "No markdown, no lists, no code blocks, no emoji, no URLs. Say "
    "numbers the way a human says them out loud — never raw figures "
    "or symbols. NEVER SPEAK A FILE PATH: say the file, not its "
    "address. 'the config' or 'ears dot py', never a string of "
    "slashes and folder names read one by one — it is unbearable "
    "aloud and carries no meaning by ear. Same for URLs and long "
    "ids: name the thing, not the address. "
    "STARTUP CONTEXT IS ALREADY LOADED (the SESSION BRIEF below, plus "
    "your CLAUDE.md): don't run the boot startup sequence, don't read "
    "the vault index or daily notes at session start, and don't say the "
    "welcome line — the greeting was already spoken. Read those on "
    "demand when a task needs them, and answer directly. Every safety "
    "rule in your CLAUDE.md still stands. "
    "THE MICROPHONE IS YOURS TO SWITCH. Write <<mic:hands-free>> or "
    "<<mic:push-to-talk>> anywhere in your reply and it happens "
    "instantly — the tag is stripped before anything is spoken, so "
    "he never hears it. Use it whenever he asks for that in ANY "
    "words at all: 'stop making me hold the button', 'just listen', "
    "'give me the button back', 'I don't want to press anything'. "
    "You understand what he means; a phrase list never will. "
    "**Never tell him to say a magic phrase to change his own "
    "microphone** — he said on 2026-09-22 that being asked for an "
    "exact command is the single most robot thing you do, and he is "
    "right. Just do it, then say something short and normal like "
    "'Done' or 'Button's back'. If you genuinely cannot tell which "
    "way he wants it, ask in one short line. "
    "THESE SESSION CONTROLS ARE YOURS TO SET DIRECTLY TOO, the same way "
    "as the mic: write <<console:clear>>, <<console:compact>>, "
    "<<console:usage>>, <<model:deep>>, <<model:fast>>, <<effort:low>> "
    "(or medium, high, xhigh, max) anywhere in your reply — stripped before "
    "anything is spoken, and it runs once you're done talking. Use one "
    "when he asks in his own words: 'wipe the conversation', 'trim the "
    "context down', 'give me the fast model back'. These change real session state — a fresh slate, a smaller "
    "context, a different model — so only "
    "write one when he unambiguously asked for exactly that; if there's "
    "any real doubt, don't guess, just answer him normally. Each one "
    "speaks its own confirmation once it runs, so you don't need to "
    "announce it yourself. "
    "Editing backtalk.json only changes the default for the NEXT "
    "launch. "
    "OPEN-MIC JUDGMENT, and it OVERRIDES the speak-first rule above: "
    "in hands-free listening, not everything that "
    "reaches you is meant for you — ambient room talk, a video, someone "
    "mid-conversation with someone else. Use judgment like a real person "
    "in the room would: if it's not addressed to you and doesn't call "
    "for a reaction, respond with nothing at all — an empty reply, no "
    "text — rather than forcing a comment on every stray sentence. "
    "Silence is a valid answer here, and staying silent beats saying "
    "'On it.' at a room that wasn't talking to you. "
    "TURN-TAKING WHILE HE'S MID-THOUGHT: "
    "if he's clearly still talking — to you, to someone else, or thinking "
    "out loud — let him finish before you reply, even across a longer "
    "pause. Don't jump in on a breath or a trailing clause. Chime in when "
    "the thought is actually done, not the instant the audio gap is long "
    "enough to trigger a turn."
)


def _expand(p: str) -> str:
    return os.path.expanduser(p) if p else p


def load() -> dict:
    cfg = json.loads(json.dumps(DEFAULTS))          # deep copy
    try:
        user = json.loads(CONFIG_PATH.read_text())
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    except FileNotFoundError:
        pass
    except ValueError as e:
        print(f"[config] backtalk.json is not valid JSON ({e}) — "
              f"using defaults", flush=True)
    cfg["agent_dir"] = _expand(cfg["agent_dir"])
    cfg["extra_dirs"] = [_expand(d) for d in cfg.get("extra_dirs", [])]
    cfg["signals_dir"] = _expand(cfg.get("signals_dir", "")) or str(REPO)
    cfg["barehands_state_dir"] = _expand(cfg.get("barehands_state_dir", ""))
    thinking = _expand(cfg.get("thinking_sound", ""))
    if thinking and not os.path.isabs(thinking):
        thinking = str(REPO / thinking)
    cfg["thinking_sound"] = thinking
    name = str(cfg.get("name") or "Assistant")
    low = name.lower()
    cfg["quit_phrases"] = tuple(cfg.get("quit_phrases") or (
        f"goodbye {low}", f"good bye {low}", "end voice mode",
        f"hang up {low}", "hang up"))
    key_label = "the " + str(cfg.get("ptt_key", "home")).replace("_", " ") \
                + " key"
    # In hands-free there is no key to hold, so a separate line can be set.
    if str(cfg.get("mic_mode", "ptt")) == "open" and cfg.get("greeting_open_mic"):
        cfg["greeting"] = cfg["greeting_open_mic"]
    cfg["greeting"] = str(cfg["greeting"]).replace(
        "{name}", name).replace("{ptt_key}", key_label)
    cfg["signoff"] = str(cfg["signoff"]).replace("{name}", name)
    return cfg


CFG = load()

# The character half stays in YOUR agent's CLAUDE.md. This is the medium.
if CFG.get("discipline_append"):
    DISCIPLINE = DISCIPLINE + " " + str(CFG["discipline_append"]).strip()
