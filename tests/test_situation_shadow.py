"""Situation Phase 3B: the SHADOW projection wired at the real send point (main.py).

The projector (situation_context.py) is called with a fresh snapshot just before brain.ask_stream, its
state moves only when the router CLAIMED the sent turn, and one metadata-only "[sitctx]" line is logged.
The note itself is never sent anywhere. These tests drive the REAL WarmBrain / router / ledger / speak_reply
over the fake SDK client; nothing here touches the live voice process."""
import ast
import asyncio
import inspect
import os
import re
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from backtalk import ledger as L
from backtalk import main as M
from backtalk import situation as S
from backtalk import situation_context as C
from tests import router_fakes as F
from tests import test_main_router as TMR
from tests.router_fakes import settle

PKG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backtalk")

LINE = re.compile(r"^\[sitctx\] projection=(baseline|delta|retraction|none)"
                  r"( fields=(mic|background|mic,background) length=\d+)? "
                  r"delivery=(claimed|unclaimed|unattributed)$")
SECRETS = ("hunter2", "sk-live-abc", "/Users/yousif", "SECRET-REPLY", "SECRET-TASK-DESC", "SECRET-FROM-STATE",
           "secret detail", "rm -rf")


class ShadowBase(TMR.Base):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        # production has a session id before the first user send (the boot warm-up's result sets it)
        self.brain.session_id = F.SID
        self.lines, self.events, self.snaps, self.notes, self.order = [], [], [], [], []
        self.mic = {"mode": "open", "btn": False}
        self.reader = S.SituationReader(S.SituationSources(          # NO observer: exactly like production's shadow reader
            mic=lambda: self.mic, zipped=lambda: False, mouth=self.mouth, brain=self.brain))
        real = self.reader.snapshot

        def spy():
            self.events.append("snapshot")
            snap = real()
            self.snaps.append(snap)
            return snap
        self.reader.snapshot = spy
        real_project = C.project

        def project_spy(*a, **k):
            p = real_project(*a, **k)
            if p.note is not None:
                self.notes.append(p.note)
            return p
        def logfn(line):
            self.lines.append(line)
            if str(line).startswith("[sitinj]"):
                self.order.append("marker")
        self.p2 = [mock.patch.dict(M.CFG),                                       # the 3C flag is restored after every test
                   mock.patch.object(M, "_SITCTX_READER", self.reader),
                   mock.patch.dict(M._SITCTX, {"state": None, "epoch": None, "session": None, "generation": None}),
                   mock.patch.dict(M._SITCTX_GEN, {"n": 0}),
                   mock.patch.dict(M._SITCTX_FAIL, {}, clear=True),
                   mock.patch.object(M, "log", logfn),
                   mock.patch.object(C, "project", project_spy)]
        for p in self.p2:
            p.start()
        M.CFG.pop(M._SITCTX_FLAG, None)                                         # default: key ABSENT (flag OFF)
        self.set_script(lambda c, m: c.feed(*F.fg_run(m["uuid"], "Hello there.")))

    async def asyncTearDown(self):
        for p in self.p2:
            p.stop()
        await super().asyncTearDown()

    def set_script(self, fn):
        def wrapped(c, m):
            self.events.append("query")
            self.order.append("query")
            fn(c, m)
        self.script = wrapped

    def flag_on(self):
        M.CFG[M._SITCTX_FLAG] = True

    def markers(self):
        return [ln for ln in self.lines if ln.startswith("[sitinj]")]

    def sitctx(self):
        return [ln for ln in self.lines if ln.startswith("[sitctx]")]

    def model_inputs(self):
        """Everything that was ever handed to the model, from the fake client."""
        return [m["message"]["content"] for c in self.clients for m in c.sent]

    async def reply_ctx(self, text, ctx):
        """speak_reply with a vault-recall block in front of the text, as production composes it."""
        fut = asyncio.get_running_loop().create_future()
        fut.set_result(ctx)
        await M.speak_reply(self.brain, self.mouth, text, vault_fut=fut)

    async def send_unclaimed(self, text="hi", feed=None, ctx=None):
        """A send whose echo never arrives (optionally with other traffic), then cancelled: an abandoned send."""
        self.set_script(lambda c, m: feed(c, m) if feed else None)
        client = self.client
        client.on_interrupt = lambda: client.feed(                            # the CLI acknowledges the interrupt
            F.result("", is_error=True, subtype="error_during_execution", terminal="aborted_streaming"))
        fut = asyncio.get_running_loop().create_future()
        fut.set_result(ctx)
        task = asyncio.create_task(M.speak_reply(self.brain, self.mouth, text, vault_fut=fut))
        await settle()
        task.cancel()
        try:
            await task
        except BaseException:
            pass
        self.set_script(lambda c, m: c.feed(*F.fg_run(m["uuid"], "Hello there.")))


# ---- A, B: where and how often ----------------------------------------------------------------

class SendPointTests(ShadowBase):
    async def test_a_fresh_snapshot_is_taken_immediately_before_the_real_send(self):
        await self.reply("hi")
        self.assertEqual(self.events, ["snapshot", "query"])
        await self.reply("again")
        self.assertEqual(self.events, ["snapshot", "query", "snapshot", "query"])

    async def test_each_send_gets_its_own_new_snapshot_never_a_reused_one(self):
        await self.reply("one")
        await asyncio.sleep(0.01)
        await self.reply("two")
        self.assertEqual(len(self.snaps), 2)
        self.assertIsNot(self.snaps[0], self.snaps[1])
        self.assertGreater(self.snaps[1].taken_at_mono, self.snaps[0].taken_at_mono)

    async def test_exactly_one_snapshot_one_projection_and_one_line_per_real_send(self):
        for i in range(3):
            await self.reply(f"turn {i}")
        self.assertEqual(len(self.snaps), 3)
        self.assertEqual(len(self.sitctx()), 3)
        self.assertEqual(len(self.clients[-1].sent), 3)

    async def test_nothing_is_projected_when_no_send_happens(self):
        fut = asyncio.get_running_loop().create_future()
        fut.set_result(None)
        with mock.patch.object(self.brain, "ask_stream", side_effect=RuntimeError("never sends")):
            await M.speak_reply(self.brain, self.mouth, "hi", vault_fut=fut)   # speak_reply reports the failure itself
        self.assertEqual(len(self.snaps), 1)                                   # the snapshot was taken at the send point...
        self.assertEqual(self.sitctx(), [])                                    # ...but no handle was ever created: no line
        self.assertIsNone(M._SITCTX["state"])

    async def test_the_projection_is_only_ever_made_from_speak_reply(self):
        src = inspect.getsource(M)
        sites = [ln.strip() for ln in src.splitlines() if "_sitctx_begin(" in ln and not ln.strip().startswith("def ")]
        self.assertEqual(len(sites), 1)
        self.assertTrue(sites[0].startswith("shadow = _sitctx_begin(brain)"))
        self.assertIn("_sitctx_begin(brain)", inspect.getsource(M.speak_reply))
        for fn in (M.stop_current_turn, M.do_stop_background, M.do_stop_everything):
            self.assertNotIn("sitctx", inspect.getsource(fn))
        self.assertNotIn("sitctx", inspect.getsource(M.StitchHold))           # held fragments never project
        self.assertNotIn("sitctx", inspect.getsource(M.release_held))
        self.assertNotIn("sitctx", inspect.getsource(M._observe_situation))  # the [sit] telemetry helper is untouched

    async def test_a_command_turn_and_the_startup_warmup_never_project(self):
        self.set_script(lambda c, m: c.feed(F.init(), F.echo(m["uuid"]), F.result("ok")))
        await self.brain.command("/effort low")
        async for _ in self.brain.ask_stream("warmup"):
            pass
        self.assertEqual(self.snaps, [])
        self.assertEqual(self.sitctx(), [])


class StitchAndReplaceTests(ShadowBase):
    async def test_a_replaced_turn_and_its_replacement_each_log_once_and_only_the_claimed_ones_are_adopted(self):
        # the first send is claimed (its echo arrived) and then cut off by a replacement
        await self.send_unclaimed("Jarvis", feed=lambda c, m: c.feed(F.init(), F.echo(m["uuid"]), F.delta("Yes")))
        self.assertEqual(len(self.sitctx()), 1)
        self.assertIn("projection=baseline", self.sitctx()[0])
        self.assertIn("delivery=claimed", self.sitctx()[0])                   # claimed: the model had it
        await self.reply("2 plus 2")
        self.assertEqual(len(self.sitctx()), 2)
        self.assertIn("projection=none delivery=claimed", self.sitctx()[1])   # already told by the first send

    async def test_a_send_abandoned_before_its_claim_is_not_adopted_and_the_replacement_re_offers_the_baseline(self):
        await self.send_unclaimed("Jarvis")                                    # never echoed
        self.assertIn("delivery=unclaimed", self.sitctx()[0])
        self.assertIsNone(M._SITCTX["state"])
        await self.reply("2 plus 2")
        self.assertIn("projection=baseline", self.sitctx()[1])                 # offered again, not suppressed
        self.assertIn("delivery=claimed", self.sitctx()[1])

    async def test_the_hold_branch_returns_before_speak_reply_is_ever_created(self):
        """handle() is a closure inside amain() (it needs the whole audio stack), so its control flow is
        proved from its AST: a held fragment leaves handle() through `if hold: ... return` BEFORE the one
        place a speak_reply task is created, so it can never reach the send point or the shadow."""
        tree = ast.parse(inspect.getsource(M))
        amain = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "amain")
        handle = next(n for n in ast.walk(amain) if isinstance(n, ast.AsyncFunctionDef) and n.name == "handle")
        holds = [n for n in handle.body if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == "hold"]
        self.assertEqual(len(holds), 1)
        hold_if = holds[0]
        self.assertIsInstance(hold_if.body[-1], ast.Return)                    # the branch always ends by returning
        inside = {n.id for st in hold_if.body for n in ast.walk(st) if isinstance(n, ast.Name)}
        for forbidden in ("speak_reply", "speak_task", "_sitctx_begin", "_sitctx_end", "ask_stream"):
            self.assertNotIn(forbidden, inside, forbidden)                     # nothing in it sends or projects
        creations = [n for n in ast.walk(handle) if isinstance(n, ast.Call)
                     and any(isinstance(a, ast.Call) and getattr(a.func, "id", None) == "speak_reply" for a in n.args)]
        self.assertEqual(len(creations), 1)                                    # ONE place a send task is made...
        self.assertGreater(creations[0].lineno, hold_if.end_lineno)            # ...and it is after the hold's return
        returns_before = [n for n in handle.body if isinstance(n, ast.If) and n.lineno < creations[0].lineno]
        self.assertIn(hold_if, returns_before)

    async def test_a_held_stitch_request_is_sent_once_as_one_replacement_never_per_fragment(self):
        clock = [0.0]
        hold = M.StitchHold(clock=lambda: clock[0])
        brain = SimpleNamespace(router=SimpleNamespace(epoch="E1"))
        sent = []

        async def handle(text, spoke_from=None, released=False):
            sent.append((text, released))
            return True
        hold.begin("first fragment", "E1")
        hold.add("second fragment")
        hold.add("third fragment")
        self.assertTrue(await M.release_held(hold, brain, handle))             # not due: nothing is sent
        self.assertEqual(sent, [])
        clock[0] = 100.0                                                       # quiet time ran out
        await M.release_held(hold, brain, handle)
        self.assertEqual(sent, [("first fragment second fragment third fragment", True)])
        await M.release_held(hold, brain, handle)                              # the hold ended: never sent twice
        self.assertEqual(len(sent), 1)
        self.assertEqual((self.snaps, self.sitctx()), ([], []))                # holding and releasing never project by themselves


# ---- D, M-P: nothing reaches the model, nothing else changes ---------------------------------

class NoModelInjectionTests(ShadowBase):
    async def test_what_the_model_receives_is_exactly_what_it_received_before_3b(self):
        await self.reply("hello there my friend")
        self.assertEqual(self.model_inputs(), ["hello there my friend"])
        self.assertGreater(len(self.notes), 0)                                 # a projection WAS produced...
        for note in self.notes:
            for sent in self.model_inputs():
                self.assertNotIn(note, sent)                                   # ...and never reached the model
                self.assertNotIn("Situation", sent)

    async def test_the_vault_context_composition_is_unchanged(self):
        ctx = "[From the vault: background only]"
        fut = asyncio.get_running_loop().create_future()
        fut.set_result(ctx)
        await M.speak_reply(self.brain, self.mouth, "what is up", vault_fut=fut)
        self.assertEqual(self.model_inputs(), [f"{ctx}\n\nwhat is up"])

    async def test_ask_stream_is_called_with_the_unchanged_text(self):
        seen = []
        real = self.brain.ask_stream

        def spy(utterance, **kw):
            seen.append(utterance)
            return real(utterance, **kw)
        self.brain.ask_stream = spy
        await self.reply("keep this exactly")
        await self.reply("and this")
        self.assertEqual(seen, ["keep this exactly", "and this"])

    async def test_no_note_reaches_the_mouth_the_ledger_or_the_logs(self):
        await self.reply("hi")
        await self.reply("again")
        self.assertGreater(len(self.notes), 0)
        for note in self.notes:
            self.assertNotIn(note, " ".join(self.mouth.chunks + self.mouth.said))
            self.assertNotIn(note, " ".join(self.lines))
            for row in self.led.turns.values():
                self.assertNotIn(note, str(row))
            for row in self.led.actions.values():
                self.assertNotIn(note, str(row))

    async def test_speak_reply_never_touches_a_projection_note(self):
        src = inspect.getsource(M.speak_reply)
        for forbidden in (".note", "Projection", "situation_context_mod", "_SITCTX[", "proj"):
            self.assertNotIn(forbidden, src, forbidden)
        tree = ast.parse(inspect.getsource(M.speak_reply).replace("\n    ", "\n", 0))
        # the shadow record is only ever READ as a direct argument of the two helpers that own it
        reads = [n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == "shadow" and isinstance(n.ctx, ast.Load)]
        owners = {"_sitctx_compose", "_sitctx_end"}
        owner_args = [a for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", None) in owners
                      for a in n.args]
        self.assertEqual(len(reads), 2)
        self.assertTrue(all(any(r is a for a in owner_args) for r in reads))
        assigns = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "sent" for t in n.targets)]
        self.assertEqual(len(assigns), 1)                                      # `sent` is composed exactly once

    async def test_the_telemetry_reader_and_its_line_are_untouched(self):
        src = inspect.getsource(M)
        self.assertIn("brain=brain, desktop=_presence_desktop, observer=log, on_failure=log)", src)
        self.assertIn("dataclasses.replace(_sit_sources, observer=None, on_failure=None)", src)
        await self.reply("hi")
        self.assertEqual([ln for ln in self.lines if ln.startswith("[sit] ")], [])   # the shadow reader emits no [sit]

    async def test_no_authority_or_runtime_module_mentions_the_shadow_or_the_projector(self):
        for name in ("addressee.py", "router.py", "brain.py", "ledger.py", "speech_policy.py",
                     "config.py", "situation.py"):
            with open(os.path.join(PKG, name), encoding="utf-8") as f:
                text = f.read()
            self.assertNotIn("sitctx", text, name)
            self.assertNotIn("situation_context", text, name)

    async def test_the_projector_stays_pure(self):
        tree = ast.parse(open(os.path.join(PKG, "situation_context.py"), encoding="utf-8").read())
        imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        imported |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        self.assertEqual(imported, {"math", "dataclasses", "datetime"})

    async def test_there_is_no_model_injection_switch_or_config(self):
        src = inspect.getsource(M)
        for word in ("situation_inject", "inject_situation", "situation_context\"", "SITCTX_INJECT"):
            self.assertNotIn(word, src)
        with open(os.path.join(PKG, "config.py"), encoding="utf-8") as f:
            self.assertNotIn("situation", f.read().lower())


# ---- E, F, G: adoption only after the router claimed the turn ----------------------------------

class AdoptionTests(ShadowBase):
    async def test_submit_only_is_not_delivery(self):
        await self.send_unclaimed("hi")                                        # query written, nothing came back
        self.assertEqual(self.clients[-1].sent[-1]["message"]["content"], "hi")
        self.assertIsNone(M._SITCTX["state"])
        self.assertIn("delivery=unclaimed", self.sitctx()[-1])

    async def test_a_stream_starting_without_the_claim_is_not_delivery(self):
        await self.send_unclaimed("hi", feed=lambda c, m: c.feed(F.init()))
        self.assertIsNone(M._SITCTX["state"])

    async def test_content_arriving_before_the_echo_is_not_delivery(self):
        await self.send_unclaimed("hi", feed=lambda c, m: c.feed(F.init(), F.delta("a first token")))
        self.assertIsNone(M._SITCTX["state"])
        self.assertIn("delivery=unclaimed", self.sitctx()[-1])

    async def test_an_interrupted_send_that_was_never_claimed_is_not_delivery(self):
        fut = asyncio.get_running_loop().create_future()
        fut.set_result(None)
        self.set_script(lambda c, m: None)
        task = asyncio.create_task(M.speak_reply(self.brain, self.mouth, "hi", vault_fut=fut))
        await settle()
        await self.brain.interrupt()
        task.cancel()
        try:
            await task
        except BaseException:
            pass
        self.assertIsNone(M._SITCTX["state"])

    async def test_a_failed_turn_before_its_claim_is_not_delivery(self):
        self.set_script(lambda c, m: c.feed(ValueError("boom")))
        try:
            await self.reply("hi")
        except Exception:
            pass
        self.assertIsNone(M._SITCTX["state"])

    async def test_the_claim_is_what_adopts_the_state(self):
        await self.reply("hi")
        st = M._SITCTX["state"]
        self.assertIsInstance(st, C.ProjectionState)
        self.assertTrue(st.baselined)
        self.assertEqual(M._SITCTX["epoch"], self.brain.router.epoch)
        self.assertIn("delivery=claimed", self.sitctx()[-1])

    async def test_a_turn_claimed_and_then_cut_off_is_adopted_because_the_model_had_the_message(self):
        await self.send_unclaimed("hi", feed=lambda c, m: c.feed(F.init(), F.echo(m["uuid"]), F.delta("Partial")))
        self.assertIsNotNone(M._SITCTX["state"])
        self.assertIn("delivery=claimed", self.sitctx()[-1])

    async def test_a_failed_delivery_makes_the_next_eligible_turn_re_offer_the_baseline(self):
        await self.send_unclaimed("one")                                       # not delivered
        await self.reply("two")                                                # delivered
        await self.reply("three")                                              # nothing new to say
        kinds = [re.match(r"\[sitctx\] projection=(\w+)", ln).group(1) for ln in self.sitctx()]
        self.assertEqual(kinds, ["baseline", "baseline", "none"])

    async def test_a_change_after_adoption_is_a_delta_with_only_the_changed_field(self):
        await self.reply("one")
        self.mic["mode"] = "ptt"
        await asyncio.sleep(0)
        await self.reply("two")
        self.assertRegex(self.sitctx()[-1], r"projection=delta fields=mic length=\d+ delivery=claimed")
        self.assertEqual(len(self.notes), 2)
        self.assertIn("mic: push-to-talk", self.notes[-1])
        self.assertNotIn("background", self.notes[-1])

    async def test_a_new_epoch_re_baselines(self):
        await self.reply("one")
        await self.reply("two")
        self.assertIn("projection=none", self.sitctx()[-1])
        await self.brain.rebuild("desync")
        await self.reply("three")
        self.assertRegex(self.sitctx()[-1], r"projection=baseline fields=mic,background length=\d+ delivery=claimed")

    async def test_the_shadow_state_is_never_adopted_for_a_projection_without_a_note(self):
        await self.reply("one")
        before = M._SITCTX["state"]
        await self.reply("two")                                                # no note
        self.assertIs(M._SITCTX["state"], before)


# ---- the decisive attribution lifecycle: identical intents must never confuse old and new handles -----

VAULT_BLOCK = ("[From the vault, background only -- use it if it helps, don't read it out loud or mention it unless "
               "he asks:\n- 09 - Resources/Voice Line (Backtalk).md (match 0.61): " + "a long shared snippet " * 12 + "]")


class AttributionLifecycleTests(ShadowBase):
    """Grounded in the REAL mechanism (real router, ledger and brain): prior-handle exclusion + the same router
    + exactly one new foreground handle + that handle's own claimed_at. h.intent is only a weak cross-check."""

    def fg_handles(self):
        return [h for h in self.brain.router.handles.values() if h.kind == "foreground"]

    async def test_an_old_claimed_handle_with_an_identical_intent_is_not_mistaken_for_a_new_unclaimed_send(self):
        await self.reply("what is two plus two")                                  # claimed and adopted
        adopted = M._SITCTX["state"]
        self.assertIsNotNone(adopted)
        self.mic["mode"] = "ptt"                                                   # so the next projection WOULD differ
        await self.send_unclaimed("what is two plus two")                          # same text, never echoed
        old, new = self.fg_handles()
        self.assertEqual(old.intent, new.intent)                                   # identical ledger/redacted intent
        self.assertEqual((old.claimed_at is not None, new.claimed_at), (True, None))
        self.assertRegex(self.sitctx()[-1], r"^\[sitctx\] projection=delta fields=mic length=\d+ delivery=unclaimed$")
        self.assertIs(M._SITCTX["state"], adopted)                                 # nothing adopted from the old handle
        self.assertEqual(dict(M._SITCTX["state"].told)["mic"], "open-mic")         # still what the model was told
        await self.reply("what is two plus two")                                   # the next claimed send re-offers it
        self.assertRegex(self.sitctx()[-1], r"^\[sitctx\] projection=delta fields=mic length=\d+ delivery=claimed$")
        self.assertEqual(dict(M._SITCTX["state"].told)["mic"], "push-to-talk")

    async def test_three_identical_sends_each_get_their_own_verdict(self):
        await self.reply("same words")
        await self.send_unclaimed("same words")
        await self.reply("same words")
        self.assertEqual([re.search(r"delivery=(\w+)", ln).group(1) for ln in self.sitctx()],
                         ["claimed", "unclaimed", "claimed"])
        self.assertEqual([h.claimed_at is not None for h in self.fg_handles()], [True, False, True])

    async def test_two_different_questions_sharing_a_vault_prefix_have_equal_intents_and_still_attribute_correctly(self):
        a, b = f"{VAULT_BLOCK}\n\nwhat is two plus two", f"{VAULT_BLOCK}\n\nwhat is the capital of france"
        self.assertNotEqual(a, b)
        self.assertEqual(L.redact(a), L.redact(b))                                 # the weak comparison cannot tell them apart
        await self.reply_ctx("what is two plus two", VAULT_BLOCK)                  # claimed, adopted
        adopted = M._SITCTX["state"]
        self.mic["mode"] = "ptt"
        await self.send_unclaimed("what is the capital of france", ctx=VAULT_BLOCK)    # different question, same intent
        old, new = self.fg_handles()
        self.assertEqual(old.intent, new.intent)
        self.assertEqual((old.claimed_at is not None, new.claimed_at), (True, None))
        self.assertIn("delivery=unclaimed", self.sitctx()[-1])                     # NOT attributed to the old claimed handle
        self.assertIs(M._SITCTX["state"], adopted)
        await self.reply_ctx("a third question entirely", VAULT_BLOCK)             # claimed: the delta is offered again
        self.assertRegex(self.sitctx()[-1], r"projection=delta fields=mic length=\d+ delivery=claimed")

    async def test_a_claimed_new_send_with_an_old_unclaimed_identical_handle_is_attributed_to_the_new_one(self):
        await self.send_unclaimed("again and again")                               # old handle: never claimed
        self.assertIsNone(M._SITCTX["state"])
        await self.reply("again and again")                                        # new handle, identical intent, claimed
        old, new = self.fg_handles()
        self.assertEqual(old.intent, new.intent)
        self.assertEqual((old.claimed_at, new.claimed_at is not None), (None, True))
        self.assertRegex(self.sitctx()[-1], r"projection=baseline fields=mic,background length=\d+ delivery=claimed")
        self.assertIsNotNone(M._SITCTX["state"])

    async def test_the_attribution_depends_on_the_key_set_not_on_the_intent_text(self):
        """With the prior-handle exclusion removed the old claimed handle WOULD pass every other check, so
        this is the property that carries the safety: it is exercised against a record that forgot the old key."""
        await self.reply("what is two plus two")
        old = self.fg_handles()[0]
        self.assertIsNotNone(old.claimed_at)
        lines = []
        with mock.patch.object(M, "log", lines.append), mock.patch.dict(M._SITCTX, {"state": None, "epoch": None}):
            rec = {"router": self.brain.router, "keys": frozenset(), "prior": None,       # forgot the old key
                   "proj": C.project(self.snaps[-1], now_mono=self.snaps[-1].taken_at_mono),
                   "identity": M._sitctx_identity(self.brain)}
            M._sitctx_end(self.brain, rec, "what is two plus two")
        self.assertIn("delivery=claimed", lines[0])                                # proves the key set is what excludes it
        lines.clear()
        with mock.patch.object(M, "log", lines.append), mock.patch.dict(M._SITCTX, {"state": None, "epoch": None}):
            rec = {"router": self.brain.router, "keys": frozenset(self.brain.router.handles), "prior": None,
                   "proj": C.project(self.snaps[-1], now_mono=self.snaps[-1].taken_at_mono),
                   "identity": M._sitctx_identity(self.brain)}
            M._sitctx_end(self.brain, rec, "what is two plus two")
        self.assertEqual(lines, [])                                                # with the key set: no new send, no line


# ---- unit tests of the attribution step ----------------------------------------------------------

class AttributionUnitTests(unittest.TestCase):
    def handle(self, uuid, kind="foreground", intent="hi", claimed=True, epoch="E1"):
        return SimpleNamespace(kind=kind, intent=intent, claimed_at=(1.0 if claimed else None), epoch=epoch, uuid=uuid)

    def brain(self, handles, router=None):
        r = router or SimpleNamespace(epoch="E1", handles=dict(handles))
        return SimpleNamespace(router=r), r

    def rec(self, router, keys=(), note=True):
        proj = C.project(snap_ok(), now_mono=1000.0)
        if not note:
            proj = C.Projection(None, proj.state)
        return {"router": router, "keys": frozenset(keys), "prior": None, "proj": proj,
                "identity": (getattr(router, "epoch", None), None, 0)}

    def run_end(self, brain, rec, sent="hi"):
        lines = []
        with mock.patch.object(M, "log", lines.append), mock.patch.dict(M._SITCTX, {"state": None, "epoch": None}), \
                mock.patch.dict(M._SITCTX_GEN, {"n": 0}):
            M._sitctx_end(brain, rec, sent)
            state = dict(M._SITCTX)
        return lines, state

    def test_exactly_one_new_claimed_foreground_handle_is_this_send_and_adopts(self):
        old = self.handle("u0")
        brain, r = self.brain({"u0": old, "u1": self.handle("u1", intent=M.ledger_mod.redact("hi"))})
        lines, st = self.run_end(brain, self.rec(r, keys=["u0"]))
        self.assertRegex(lines[0], r"delivery=claimed$")
        self.assertIsNotNone(st["state"])
        self.assertEqual(st["epoch"], "E1")

    def test_an_unclaimed_new_handle_logs_unclaimed_and_adopts_nothing(self):
        brain, r = self.brain({"u1": self.handle("u1", intent="hi", claimed=False)})
        lines, st = self.run_end(brain, self.rec(r))
        self.assertRegex(lines[0], r"delivery=unclaimed$")
        self.assertIsNone(st["state"])

    def test_two_new_foreground_handles_are_unattributed_and_adopt_nothing(self):
        brain, r = self.brain({"u1": self.handle("u1"), "u2": self.handle("u2")})
        lines, st = self.run_end(brain, self.rec(r))
        self.assertRegex(lines[0], r"delivery=unattributed$")
        self.assertIsNone(st["state"])

    def test_a_handle_whose_text_is_not_the_text_sent_is_unattributed(self):
        brain, r = self.brain({"u1": self.handle("u1", intent="something else")})
        lines, st = self.run_end(brain, self.rec(r))
        self.assertRegex(lines[0], r"delivery=unattributed$")
        self.assertIsNone(st["state"])

    def test_command_handles_are_not_this_send(self):
        brain, r = self.brain({"u1": self.handle("u1", kind="command")})
        lines, st = self.run_end(brain, self.rec(r))
        self.assertEqual(lines, [])
        self.assertIsNone(st["state"])

    def test_no_new_handle_means_no_send_means_no_line(self):
        brain, r = self.brain({"u0": self.handle("u0")})
        lines, st = self.run_end(brain, self.rec(r, keys=["u0"]))
        self.assertEqual(lines, [])

    def test_a_rebuilt_router_counts_every_handle_in_the_new_router_as_new_but_adopts_nothing(self):
        old_router = SimpleNamespace(epoch="E0", handles={"u0": self.handle("u0", epoch="E0")})
        new_router = SimpleNamespace(epoch="E1", handles={"u9": self.handle("u9", intent=M.ledger_mod.redact("hi"))})
        lines, st = self.run_end(SimpleNamespace(router=new_router), self.rec(old_router, keys=["u0"]))
        self.assertEqual(len(lines), 1)                                        # the new router's handle WAS found (attribution)
        self.assertRegex(lines[0], r"delivery=unattributed$")                  # ...but it is a different context
        self.assertIsNone(st["state"])                                         # so nothing is adopted

    def test_a_key_seen_on_the_old_router_does_not_hide_a_handle_on_the_new_router(self):
        old_router = SimpleNamespace(epoch="E0", handles={"same": self.handle("same", epoch="E0")})
        new_router = SimpleNamespace(epoch="E1", handles={"same": self.handle("same", intent=M.ledger_mod.redact("hi"))})
        lines, st = self.run_end(SimpleNamespace(router=new_router), self.rec(old_router, keys=["same"]))
        self.assertEqual(len(lines), 1)                                        # found: a different router, every handle is new
        self.assertRegex(lines[0], r"delivery=unattributed$")                  # but not the context the projection described
        self.assertIsNone(st["state"])

    def test_on_the_same_router_a_key_seen_before_the_send_is_not_this_send(self):
        r = SimpleNamespace(epoch="E1", handles={"same": self.handle("same", intent="hi")})
        lines, st = self.run_end(SimpleNamespace(router=r), self.rec(r, keys=["same"]))
        self.assertEqual(lines, [])                                            # nothing new was created: no send

    def test_a_missing_record_or_projection_or_router_does_nothing(self):
        brain, r = self.brain({"u1": self.handle("u1")})
        for rec in (None, {"router": r, "keys": frozenset(), "prior": None, "proj": None}):
            self.assertEqual(self.run_end(brain, rec)[0], [])
        self.assertEqual(self.run_end(SimpleNamespace(router=None), self.rec(r))[0], [])

    def test_a_handle_with_no_note_is_logged_but_adopts_nothing(self):
        brain, r = self.brain({"u1": self.handle("u1", intent="hi")})
        lines, st = self.run_end(brain, self.rec(r, note=False))
        self.assertEqual(lines, ["[sitctx] projection=none delivery=claimed"])
        self.assertIsNone(st["state"])

    def test_a_corrupt_router_state_never_raises(self):
        class Bad:
            epoch = "E1"
            @property
            def handles(self):
                raise RuntimeError("secret detail")
        rec = self.rec(SimpleNamespace(epoch="E1", handles={}))
        lines = []
        with mock.patch.object(M, "log", lines.append), mock.patch.dict(M._SITCTX_FAIL, {}, clear=True):
            M._sitctx_end(SimpleNamespace(router=Bad()), rec, "hi")
        self.assertEqual(lines, ["[sitctx] projection_failure category=delivery"])


def snap_ok(mic="open", wall=1791315660.0):
    """A real, fresh snapshot with a ledger-backed zero background count."""
    router = SimpleNamespace(epoch="E1", state="OPEN", usable=True, handles={}, active_handle=lambda: None)
    brain = SimpleNamespace(router=router, session_id="s", live_background=lambda: [])
    return S.SituationReader(S.SituationSources(
        mic=lambda: {"mode": mic, "btn": False}, zipped=lambda: False, mouth=SimpleNamespace(busy=False),
        brain=brain, clock=lambda: 1000.0, wall_clock=lambda: wall)).snapshot()


# ---- the log line ----------------------------------------------------------------------------------

class LineFormatTests(unittest.TestCase):
    def proj(self, mic="open", prior=None):
        return C.project(snap_ok(mic), now_mono=1000.0, state=prior)

    def test_the_exact_baseline_shape(self):
        p = self.proj()
        self.assertEqual(M._sitctx_line(None, p, "claimed"),
                         f"[sitctx] projection=baseline fields=mic,background length={len(p.note)} delivery=claimed")

    def test_the_exact_delta_shape(self):
        base = self.proj()
        p = self.proj("ptt", prior=base.state)
        self.assertEqual(M._sitctx_line(base.state, p, "claimed"),
                         f"[sitctx] projection=delta fields=mic length={len(p.note)} delivery=claimed")

    def test_the_exact_retraction_shape(self):
        base = self.proj()
        gone = C.project(snap_ok(), now_mono=2000.0, state=base.state)              # both stale
        self.assertEqual(M._sitctx_line(base.state, gone, "claimed"),
                         f"[sitctx] projection=retraction fields=mic,background length={len(gone.note)} delivery=claimed")

    def test_the_exact_none_shape(self):
        base = self.proj()
        none = self.proj(prior=base.state)
        self.assertEqual(M._sitctx_line(base.state, none, "unclaimed"), "[sitctx] projection=none delivery=unclaimed")

    def test_every_possible_line_matches_the_strict_shape_and_carries_no_value(self):
        base = self.proj()
        cases = [(None, self.proj()), (base.state, self.proj("ptt", prior=base.state)),
                 (base.state, C.project(snap_ok(), now_mono=2000.0, state=base.state)),
                 (base.state, self.proj(prior=base.state))]
        for prior, p in cases:
            for d in ("claimed", "unclaimed", "unattributed"):
                line = M._sitctx_line(prior, p, d)
                self.assertRegex(line, LINE)
                for word in ("open-mic", "push-to-talk", "background agents", "PM", "AM", "2026", "Tue"):
                    self.assertNotIn(word, line)

    def test_the_field_names_come_only_from_the_fixed_allowlist(self):
        self.assertEqual(M._SITCTX_FIELDS, ("mic", "background"))
        evil = C.ProjectionState(True, (("mic", "SECRET-FROM-STATE"), ("evil", "payload"), ("/Users/x", "y")))
        p = self.proj("ptt", prior=evil)
        line = M._sitctx_line(evil, p, "claimed")
        self.assertRegex(line, LINE)
        self.assertNotIn("SECRET", line)
        self.assertNotIn("evil", line)


# ---- H, I, J: log safety ----------------------------------------------------------------------------

class LogSafetyTests(ShadowBase):
    async def test_the_line_carries_only_the_approved_metadata(self):
        await self.reply("please do hunter2 sk-live-abc /Users/yousif/x")
        await self.reply("rm -rf /")
        lines = self.sitctx()
        self.assertEqual(len(lines), 2)
        for ln in lines:
            self.assertRegex(ln, LINE)

    async def test_no_id_value_text_timestamp_or_secret_reaches_the_line(self):
        self.set_script(lambda c, m: c.feed(*F.fg_run(m["uuid"], "SECRET-REPLY is here.")))
        await self.reply("tell me hunter2 /Users/yousif sk-live-abc")
        text = " ".join(self.sitctx())
        r = self.brain.router
        handle_ids = [h.turn_id for h in r.handles.values()] + [h.uuid for h in r.handles.values()] + \
                     [h.uuid[:8] for h in r.handles.values()]
        for bad in SECRETS + (F.SID, r.epoch, "open-mic", "push-to-talk", "Tue", "2026", "background agents") + tuple(handle_ids):
            self.assertNotIn(bad, text, bad)
        self.assertNotRegex(text, r"\d{1,2}:\d\d")                              # no timestamps
        self.assertNotIn("/", text)

    async def test_a_poisoned_background_task_cannot_leak_through_the_line(self):
        def script(c, m):
            c.feed(F.init(), F.echo(m["uuid"]), *F.launch_bg("tk-secret", "toolu_s", "SECRET-TASK-DESC rm -rf"),
                   F.delta("Started."), F.asst("x"), F.result("x"))
        self.set_script(script)
        await self.reply("run it")
        await self.reply("and again")
        text = " ".join(self.lines)
        for ln in self.sitctx():
            self.assertRegex(ln, LINE)
        self.assertNotIn("SECRET-TASK-DESC", " ".join(self.sitctx()))
        self.assertTrue(all("SECRET-TASK-DESC" not in n for n in self.notes))
        self.assertEqual(self.sitctx()[-1].count("background"), 1)              # the field NAME only

    async def test_a_poisoned_caller_state_cannot_leak_through_the_line_or_the_note(self):
        epoch = self.brain.router.epoch
        M._SITCTX["epoch"], M._SITCTX["session"], M._SITCTX["generation"] = epoch, F.SID, M._SITCTX_GEN["n"]
        M._SITCTX["state"] = C.ProjectionState(True, (("mic", "SECRET-FROM-STATE rm -rf"), ("zzz", "/Users/yousif")))
        await self.reply("hi")
        for ln in self.sitctx():
            self.assertRegex(ln, LINE)
            self.assertNotIn("SECRET", ln)
        for note in self.notes:
            self.assertNotIn("SECRET", note)
            self.assertNotIn("/Users", note)


# ---- K, L: failures never touch the turn --------------------------------------------------------------

class FailureTests(ShadowBase):
    async def test_a_projector_failure_does_not_break_the_turn_and_logs_a_fixed_line_once(self):
        def boom(*a, **k):
            raise RuntimeError("secret detail /Users/yousif hunter2")
        with mock.patch.object(C, "project", boom):
            await self.reply("one")
            await self.reply("two")
        self.assertEqual(self.mouth.chunks, ["Hello there.", "Hello there."])      # both turns spoke normally
        fails = [ln for ln in self.lines if "projection_failure" in ln]
        self.assertEqual(fails, ["[sitctx] projection_failure category=projection"])   # rate-limited: once
        for ln in self.lines:
            self.assertNotIn("secret detail", ln)
            self.assertNotIn("hunter2", ln)
        self.assertEqual(self.sitctx().count("[sitctx] projection_failure category=projection"), 1)
        self.assertEqual(self.model_inputs(), ["one", "two"])

    async def test_a_failure_line_is_offered_again_after_the_interval(self):
        def boom(*a, **k):
            raise RuntimeError("x")
        with mock.patch.object(C, "project", boom):
            await self.reply("one")
            M._SITCTX_FAIL["projection"] = time.monotonic() - (M._SITCTX_FAIL_EVERY_S + 1)
            await self.reply("two")
        self.assertEqual(len([ln for ln in self.lines if "projection_failure" in ln]), 2)

    async def test_a_snapshot_failure_does_not_break_the_turn(self):
        self.reader.snapshot = mock.Mock(side_effect=RuntimeError("secret detail"))
        await self.reply("hi")
        self.assertEqual(self.mouth.chunks, ["Hello there."])
        self.assertEqual([ln for ln in self.lines if "projection_failure" in ln],
                         ["[sitctx] projection_failure category=projection"])
        self.assertEqual(self.model_inputs(), ["hi"])

    async def test_failed_projection_adopts_nothing(self):
        def boom(*a, **k):
            raise RuntimeError("x")
        with mock.patch.object(C, "project", boom):
            await self.reply("hi")
        self.assertIsNone(M._SITCTX["state"])

    async def test_a_logging_failure_does_not_break_the_turn(self):
        real = self.lines.append

        def flaky(line):
            if line.startswith("[sitctx]"):
                raise OSError("disk full secret detail")
            real(line)
        with mock.patch.object(M, "log", flaky):
            await self.reply("hi")
            await self.reply("again")
        self.assertEqual(self.mouth.chunks, ["Hello there.", "Hello there."])
        self.assertEqual(self.model_inputs(), ["hi", "again"])

    async def test_the_shadow_hooks_never_raise_into_the_voice_loop(self):
        brain = SimpleNamespace(router=object())                                # nothing usable at all
        self.assertIsNone(M._sitctx_begin(brain))
        M._sitctx_end(brain, {"router": None, "keys": frozenset(), "prior": None, "proj": None}, "x")
        M._sitctx_end(brain, None, "x")
        M._sitctx_end(None, {"router": 1}, "x")

    async def test_with_no_shadow_reader_wired_everything_is_a_no_op(self):
        with mock.patch.object(M, "_SITCTX_READER", None):
            await self.reply("hi")
        self.assertEqual(self.sitctx(), [])
        self.assertEqual(self.model_inputs(), ["hi"])
        self.assertEqual(self.snaps, [])


class FailureRateLimitTests(unittest.TestCase):
    def test_fixed_categories_are_rate_limited_per_category(self):
        lines = []
        with mock.patch.object(M, "log", lines.append), mock.patch.dict(M._SITCTX_FAIL, {}, clear=True):
            for _ in range(5):
                M._sitctx_fail("projection")
            M._sitctx_fail("delivery")
        self.assertEqual(lines, ["[sitctx] projection_failure category=projection",
                                 "[sitctx] projection_failure category=delivery"])

    def test_a_failing_log_inside_the_failure_path_is_swallowed(self):
        def bad(_l):
            raise OSError("x")
        with mock.patch.object(M, "log", bad), mock.patch.dict(M._SITCTX_FAIL, {}, clear=True):
            M._sitctx_fail("projection")


# ---- scope: nothing outside main.py's shadow wiring changed -------------------------------------------

class ScopeTests(unittest.TestCase):
    FUNCS = ("_sitctx_fail", "_sitctx_begin", "_sitctx_line", "_sitctx_end", "_sitctx_identity",
             "_sitctx_invalidate", "_sitctx_same_context", "_sitctx_note_for", "_sitctx_compose", "_sitctx_marker")

    def idents(self):
        names = set()
        for fn in self.FUNCS:
            tree = ast.parse(inspect.getsource(getattr(M, fn)))
            for n in ast.walk(tree):
                if isinstance(n, ast.Name):
                    names.add(n.id)
                elif isinstance(n, ast.Attribute):
                    names.add(n.attr)
        return names

    def test_the_shadow_code_has_no_route_to_the_model_the_voice_the_config_or_the_gates(self):
        names = self.idents()
        for word in ("ask_stream", "query", "send", "say", "say_chunk", "CFG", "DISCIPLINE", "system_prompt",
                     "ClaudeSDKClient", "_ADDRESSEE",
                     "interrupt", "stop_task", "shut_up", "set_state", "mouth", "brief", "vaultrecall"):
            self.assertNotIn(word, names, word)

    def test_the_shadow_only_reads_public_router_state_and_writes_only_its_own_module_state(self):
        names = self.idents()
        self.assertTrue({"handles", "intent", "epoch", "router"} <= names)
        src = inspect.getsource(M._sitctx_end)
        self.assertIn('getattr(h, "claimed_at", None)', src)                    # the router's own claim marker
        self.assertIn('getattr(h, "kind", None) == "foreground"', src)
        self.assertFalse({n for n in names if n.startswith("_") and n not in
                          {"_SITCTX", "_SITCTX_READER", "_SITCTX_FAIL", "_SITCTX_FAIL_EVERY_S", "_SITCTX_FIELDS",
                           "_SITCTX_GEN", "_SITCTX_NOTE_MAX", "_sitctx_fail", "_sitctx_begin", "_sitctx_line", "_sitctx_end",
                           "_sitctx_identity", "_sitctx_invalidate", "_sitctx_same_context", "_sitctx_inject_enabled",
                           "_sitctx_note_for", "_sitctx_compose", "_sitctx_marker"}})

    def test_the_shadow_adds_no_thread_timer_or_task(self):
        names = self.idents()
        for word in ("Thread", "Timer", "create_task", "run_in_executor", "sleep", "call_later", "ensure_future",
                     "to_thread"):
            self.assertNotIn(word, names, word)


# =====================================================================================================
# PRE-3C HARDENING: context invalidation (/clear, /compact, session change) and adoption identity.
# =====================================================================================================

def console_branches():
    """The nested `_run_console_inner` of amain(), and its `if verb == "<name>":` branches by name."""
    tree = ast.parse(inspect.getsource(M))
    amain = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "amain")
    fn = next(n for n in ast.walk(amain) if isinstance(n, ast.AsyncFunctionDef) and n.name == "_run_console_inner")
    out = {}
    for n in ast.walk(fn):
        if (isinstance(n, ast.If) and isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Name)
                and n.test.left.id == "verb" and isinstance(n.test.ops[0], ast.Eq)
                and isinstance(n.test.comparators[0], ast.Constant)):
            out[n.test.comparators[0].value] = n
    return fn, out


class ContextInvalidationTests(ShadowBase):
    """/clear and /compact change what the model can still see WITHOUT changing the router epoch (a /clear
    starts a new session id inside the same epoch, FACT from the production ledger), so the shadow state is
    keyed by (epoch, session id, generation) and each of them bumps the generation."""

    async def test_the_first_send_is_a_baseline_and_an_unchanged_repeat_is_none(self):
        await self.reply("one")
        await self.reply("two")
        self.assertEqual([re.search(r"projection=(\w+)", ln).group(1) for ln in self.sitctx()], ["baseline", "none"])

    async def test_the_adopted_state_records_epoch_session_and_generation(self):
        await self.reply("one")
        self.assertEqual((M._SITCTX["epoch"], M._SITCTX["session"], M._SITCTX["generation"]),
                         (self.brain.router.epoch, F.SID, 0))

    async def test_a_clear_or_compact_invalidates_the_told_state_for_the_next_real_send(self):
        await self.reply("one")
        await self.reply("two")
        epoch, session = self.brain.router.epoch, self.brain.session_id
        for expected_generation in (1, 2):                                        # a /clear, then a /compact
            M._sitctx_invalidate()                                                # exactly what the console calls
            self.assertEqual(M._SITCTX_GEN["n"], expected_generation)
            await self.reply("after")
            self.assertRegex(self.sitctx()[-1], r"projection=baseline fields=mic,background length=\d+ delivery=claimed")
            self.assertEqual(M._SITCTX["generation"], expected_generation)
            await self.reply("again")
            self.assertIn("projection=none", self.sitctx()[-1])                  # told again, so quiet again
        self.assertEqual((self.brain.router.epoch, self.brain.session_id), (epoch, session))   # NO epoch bump, NO new session

    async def test_the_generation_counter_only_ever_goes_up_and_only_the_invalidator_writes_it(self):
        for n in range(1, 6):
            M._sitctx_invalidate()
            self.assertEqual(M._SITCTX_GEN["n"], n)
        tree = ast.parse(inspect.getsource(M))
        writers = set()
        for fn in ast.walk(tree):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for n in ast.walk(fn):
                    tgt = [n.target] if isinstance(n, ast.AugAssign) else (n.targets if isinstance(n, ast.Assign) else [])
                    for t in tgt:
                        if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and t.value.id == "_SITCTX_GEN":
                            writers.add((fn.name, type(n).__name__, type(n.op).__name__ if isinstance(n, ast.AugAssign) else "-"))
        self.assertEqual(writers, {("_sitctx_invalidate", "AugAssign", "Add")})  # +1 and nothing else, nowhere else

    async def test_the_console_bumps_the_generation_before_the_command_so_a_failing_command_still_counts(self):
        fn, branches = console_branches()
        self.assertEqual(sorted(k for k in branches if k in ("clear", "compact")), ["clear", "compact"])
        for verb, branch in (("clear", branches["clear"]), ("compact", branches["compact"])):
            def call_name(stmt):
                for n in ast.walk(stmt):
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                        return n.func.id
                return None
            bump = [i for i, st in enumerate(branch.body) if isinstance(st, ast.Expr) and call_name(st) == "_sitctx_invalidate"]
            cmd = [i for i, st in enumerate(branch.body)
                   if any(isinstance(n, ast.Await) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
                          and n.value.func.attr == "command" for n in ast.walk(st))]
            self.assertEqual((len(bump), len(cmd)), (1, 1), verb)                # one unconditional bump, one command
            self.assertLess(bump[0], cmd[0], verb)                              # the bump comes first, as a direct statement
        uses = [n for n in ast.walk(ast.parse(inspect.getsource(M)))
                if isinstance(n, ast.Name) and n.id == "_sitctx_invalidate"]
        self.assertEqual(len(uses), 2)                                         # exactly the clear and compact branches
        wrapper = inspect.getsource(M)
        self.assertIn("log(f\"[console] {verb} failed: {e}\")", wrapper)        # a failure is caught OUTSIDE, after the bump

    async def test_a_changed_session_id_forces_a_new_baseline_even_without_a_generation_bump(self):
        await self.reply("one")
        await self.reply("two")
        epoch = self.brain.router.epoch
        with mock.patch.object(F, "SID", "sess-2"):                              # the CLI's reply to /clear carries a NEW session id
            def cmd(c, m):
                c.feed(F.init(), F.asst("Cleared."), F.echo(m["uuid"], "<command-name>/clear"),
                       F.result("Cleared.", num_turns=0))
            self.set_script(cmd)
            await self.brain.command("/clear")
            self.set_script(lambda c, m: c.feed(*F.fg_run(m["uuid"], "Hello there.")))
            self.assertEqual((self.brain.session_id, self.brain.router.epoch), ("sess-2", epoch))   # same epoch, new session
            self.assertEqual(M._SITCTX_GEN["n"], 0)                              # and nothing bumped the generation
            await self.reply("three")
            await self.reply("four")
        self.assertRegex(self.sitctx()[-2], r"projection=baseline fields=mic,background length=\d+ delivery=claimed")
        self.assertIn("projection=none", self.sitctx()[-1])
        self.assertEqual(M._SITCTX["session"], "sess-2")

    async def test_same_epoch_session_and_generation_keep_delta_and_none_behavior(self):
        await self.reply("one")
        self.mic["mode"] = "ptt"
        await self.reply("two")
        await self.reply("three")
        self.assertEqual([re.search(r"projection=(\w+)", ln).group(1) for ln in self.sitctx()], ["baseline", "delta", "none"])

    async def test_an_epoch_change_forces_a_fresh_baseline(self):
        await self.reply("one")
        await self.brain.rebuild("desync")
        await self.reply("two")
        self.assertRegex(self.sitctx()[-1], r"projection=baseline fields=mic,background length=\d+ delivery=claimed")

    async def test_the_shadow_can_never_write_into_the_turn_text_or_last_turn(self):
        tree = ast.parse(inspect.getsource(M))
        amain = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "amain")
        handle = next(n for n in ast.walk(amain) if isinstance(n, ast.AsyncFunctionDef) and n.name == "handle")
        names = {n.id for n in ast.walk(handle) if isinstance(n, ast.Name)} | \
                {n.attr for n in ast.walk(handle) if isinstance(n, ast.Attribute)}
        for forbidden in ("_SITCTX", "_SITCTX_GEN", "_SITCTX_READER", "situation_context_mod", "_sitctx_begin", "_sitctx_end",
                          "_sitctx_line", "_sitctx_identity", "_sitctx_same_context"):
            self.assertNotIn(forbidden, names, forbidden)                       # handle() owns _LAST_TURN and the stitch text
        self.assertIn("_LAST_TURN", {n.id for n in ast.walk(handle) if isinstance(n, ast.Name)})


class SendTimeConsistencyTests(ShadowBase):
    """The projection and its adoption must describe the same Claude context the handle belongs to."""

    async def test_the_session_is_settled_once_before_the_snapshot_and_the_send(self):
        real = self.brain.reset_turn

        async def spy(*a, **k):
            self.events.append("reset_turn")
            return await real(*a, **k)
        self.brain.reset_turn = spy
        await self.reply("hi")
        self.assertEqual(self.events, ["reset_turn", "snapshot", "query"])

    async def test_without_a_shadow_reader_the_send_path_is_exactly_what_it_was_before(self):
        seen = []
        real = self.brain.reset_turn

        async def spy(*a, **k):
            seen.append(1)
            return await real(*a, **k)
        self.brain.reset_turn = spy
        with mock.patch.object(M, "_SITCTX_READER", None):
            await self.reply("hi")
        self.assertEqual(seen, [])                                              # speak_reply did not add a reset_turn call
        self.assertEqual(self.model_inputs(), ["hi"])

    async def test_a_rebuild_due_before_the_send_is_settled_first_so_the_projection_and_the_handle_agree(self):
        await self.reply("one")
        first_epoch = self.brain.router.epoch
        self.brain._need_rebuild = "stream_lost"                                # a stream loss during handle()'s earlier yields
        await self.reply("two")
        self.assertEqual(len(self.clients), 2)                                  # exactly one rebuild
        self.assertNotEqual(self.brain.router.epoch, first_epoch)
        self.assertRegex(self.sitctx()[-1], r"projection=baseline fields=mic,background length=\d+ delivery=claimed")
        self.assertEqual(M._SITCTX["epoch"], self.brain.router.epoch)           # adopted under the NEW identity

    async def test_a_rebuild_during_the_send_is_not_adopted_and_the_next_send_re_baselines(self):
        await self.reply("one")
        adopted = M._SITCTX["state"]
        self.mic["mode"] = "ptt"                                                 # so the second projection is a delta
        real_begin = M._sitctx_begin

        def begin_then_lose_the_stream(brain):
            rec = real_begin(brain)
            brain._need_rebuild = "stream_lost"                                 # lands AFTER the identity was captured
            return rec
        with mock.patch.object(M, "_sitctx_begin", begin_then_lose_the_stream):
            await self.reply("two")
        self.assertEqual(len(self.clients), 2)                                  # the send itself rebuilt the session
        self.assertRegex(self.sitctx()[-1], r"projection=delta fields=mic length=\d+ delivery=unattributed")   # claimed, but not adopted
        self.assertIs(M._SITCTX["state"], adopted)
        await self.reply("three")
        self.assertRegex(self.sitctx()[-1], r"projection=baseline fields=mic,background length=\d+ delivery=claimed")

    async def test_a_session_id_first_assigned_during_the_send_is_not_adopted_fail_closed(self):
        self.brain.session_id = None                                            # no conversation identified yet
        await self.reply("one")                                                 # the result assigns one mid-send
        self.assertIn("delivery=unattributed", self.sitctx()[-1])
        self.assertIsNone(M._SITCTX["state"])
        await self.reply("two")                                                 # now stable: baseline again, adopted
        self.assertRegex(self.sitctx()[-1], r"projection=baseline fields=mic,background length=\d+ delivery=claimed")
        self.assertIsNotNone(M._SITCTX["state"])

    async def test_a_stream_failure_after_a_successful_claim_still_counts_as_delivered(self):
        self.set_script(lambda c, m: c.feed(F.init(), F.echo(m["uuid"]), F.delta("Partial."), ValueError("boom")))
        try:
            await self.reply("one")
        except Exception:
            pass
        self.assertRegex(self.sitctx()[-1], r"projection=baseline fields=mic,background length=\d+ delivery=claimed")
        self.assertIsNotNone(M._SITCTX["state"])                                # the model had the message: claimed means delivered
        self.set_script(lambda c, m: c.feed(*F.fg_run(m["uuid"], "Hello there.")))
        await self.reply("two")                                                 # the lost stream forces a rebuild: new epoch
        self.assertRegex(self.sitctx()[-1], r"projection=baseline fields=mic,background length=\d+ delivery=claimed")


class AdoptionGuardUnitTests(unittest.TestCase):
    """Router instance, epoch, session id and generation must ALL still match for a claimed send to be adopted."""

    def setUp(self):
        self.router = SimpleNamespace(epoch="E1", handles={})
        self.h = SimpleNamespace(kind="foreground", intent=M.ledger_mod.redact("hi"), claimed_at=1.0, epoch="E1", uuid="u1")
        self.router.handles["u1"] = self.h
        self.brain = SimpleNamespace(router=self.router, session_id="S1")
        self.proj = C.project(snap_ok(), now_mono=1000.0)

    def rec(self, **over):
        rec = {"router": self.router, "keys": frozenset(), "prior": None, "proj": self.proj, "identity": ("E1", "S1", 0)}
        rec.update(over)
        return rec

    def end(self, brain=None, rec=None, generation=0):
        lines = []
        with mock.patch.object(M, "log", lines.append), mock.patch.dict(M._SITCTX, {"state": None, "epoch": None,
                                                                                      "session": None, "generation": None}), \
                mock.patch.dict(M._SITCTX_GEN, {"n": generation}):
            M._sitctx_end(brain or self.brain, rec or self.rec(), "hi")
            state = dict(M._SITCTX)
        return lines, state

    def test_a_matching_router_epoch_session_and_generation_is_adopted(self):
        lines, st = self.end()
        self.assertRegex(lines[0], r"delivery=claimed$")
        self.assertEqual((st["epoch"], st["session"], st["generation"]), ("E1", "S1", 0))
        self.assertIs(st["state"], self.proj.state)

    def test_a_different_router_instance_is_not_adopted_even_with_the_same_epoch(self):
        other = SimpleNamespace(epoch="E1", handles=dict(self.router.handles))
        lines, st = self.end(brain=SimpleNamespace(router=other, session_id="S1"))
        self.assertRegex(lines[0], r"delivery=unattributed$")
        self.assertIsNone(st["state"])

    def test_a_different_session_id_is_not_adopted(self):
        lines, st = self.end(brain=SimpleNamespace(router=self.router, session_id="S2"))
        self.assertRegex(lines[0], r"delivery=unattributed$")
        self.assertIsNone(st["state"])

    def test_a_different_generation_is_not_adopted(self):
        lines, st = self.end(generation=1)
        self.assertRegex(lines[0], r"delivery=unattributed$")
        self.assertIsNone(st["state"])

    def test_a_handle_from_a_different_epoch_is_not_adopted(self):
        self.h.epoch = "E2"
        lines, st = self.end()
        self.assertRegex(lines[0], r"delivery=unattributed$")
        self.assertIsNone(st["state"])

    def test_each_mismatch_alone_is_enough_and_each_match_alone_is_not(self):
        for label, kw in (("router", {"brain": SimpleNamespace(router=SimpleNamespace(epoch="E1", handles=self.router.handles), session_id="S1")}),
                          ("session", {"brain": SimpleNamespace(router=self.router, session_id=None)}),
                          ("generation", {"generation": 7})):
            lines, st = self.end(**kw)
            self.assertIsNone(st["state"], label)
        self.assertTrue(M._sitctx_same_context(self.brain, self.rec(), self.h))
        for bad in ({"router": object()}, {"identity": ("E9", "S1", 0)}, {"identity": ("E1", "S9", 0)}, {"identity": ("E1", "S1", 5)}):
            self.assertFalse(M._sitctx_same_context(self.brain, self.rec(**bad), self.h), bad)

    def test_an_unclaimed_send_stays_unclaimed_and_adopts_nothing_whatever_the_identity(self):
        self.h.claimed_at = None
        lines, st = self.end()
        self.assertRegex(lines[0], r"delivery=unclaimed$")
        self.assertIsNone(st["state"])
        lines, st = self.end(generation=1)
        self.assertRegex(lines[0], r"delivery=unclaimed$")                       # a mismatch never upgrades or hides it

    def test_a_record_without_an_identity_fails_closed_without_raising(self):
        rec = self.rec()
        del rec["identity"]
        lines = []
        with mock.patch.object(M, "log", lines.append), mock.patch.dict(M._SITCTX_FAIL, {}, clear=True), \
                mock.patch.dict(M._SITCTX, {"state": None, "epoch": None, "session": None, "generation": None}):
            M._sitctx_end(self.brain, rec, "hi")
            self.assertIsNone(M._SITCTX["state"])
        self.assertEqual(lines, ["[sitctx] projection_failure category=delivery"])

    def test_the_identity_helper_reads_only_and_never_raises(self):
        self.assertEqual(M._sitctx_identity(self.brain), ("E1", "S1", M._SITCTX_GEN["n"]))
        self.assertEqual(M._sitctx_identity(SimpleNamespace()), (None, None, M._SITCTX_GEN["n"]))
        self.assertEqual(M._sitctx_identity(None), (None, None, M._SITCTX_GEN["n"]))


# =====================================================================================================
# PHASE 3C: default-OFF injection of the situation note into the model input (speak_reply only).
# =====================================================================================================

SEP = "\n\n"
MARKER = re.compile(r"^\[sitinj\] flag=(on|off) injected=(yes|no) note_chars=\d+ base_chars=\d+ final_chars=\d+$")


def marker_for(flag, note, base, final):
    """The exact marker line a send must log, built from the real strings (never a guessed number)."""
    n = len(note) if note else 0
    return (f"[sitinj] flag={flag} injected={'yes' if note else 'no'} note_chars={n} "
            f"base_chars={len(base)} final_chars={len(final)}")


class InjectionFlagTests(unittest.TestCase):
    def test_the_key_and_its_default(self):
        self.assertEqual(M._SITCTX_FLAG, "situation_context_inject")
        import backtalk.config as cfgmod
        self.assertNotIn(M._SITCTX_FLAG, cfgmod.DEFAULTS)                    # config.py was not touched: absent means OFF
        with mock.patch.dict(M.CFG):
            M.CFG.pop(M._SITCTX_FLAG, None)
            self.assertIs(M._sitctx_inject_enabled(), False)

    def test_only_the_literal_true_turns_it_on(self):
        for value, want in ((True, True), (False, False), (None, False), (1, False), (0, False), ("true", False),
                            ("yes", False), ("on", False), ("1", False), ([], False), ({}, False), ([True], False)):
            with mock.patch.dict(M.CFG, {M._SITCTX_FLAG: value}):
                self.assertIs(M._sitctx_inject_enabled(), want, repr(value))

    def test_an_unreadable_config_is_off(self):
        class Bad(dict):
            def get(self, *a, **k):
                raise RuntimeError("config unreadable")
        with mock.patch.object(M, "CFG", Bad()):
            self.assertIs(M._sitctx_inject_enabled(), False)

    def test_the_note_cap_equals_the_projectors_own_cap(self):
        self.assertEqual(M._SITCTX_NOTE_MAX, C.MAX_CHARS)
        self.assertEqual(M._SITCTX_NOTE_MAX, 400)


class InjectionComposeUnitTests(unittest.TestCase):
    NOTE = "[Situation as of 3:52 PM: mic: push-to-talk.]"

    def rec(self, note=NOTE):
        return {"proj": C.Projection(note, C.ProjectionState())}

    def on(self):
        return mock.patch.dict(M.CFG, {M._SITCTX_FLAG: True})

    def off(self):
        p = mock.patch.dict(M.CFG)
        p.start()
        M.CFG.pop(M._SITCTX_FLAG, None)
        self.addCleanup(p.stop)

    def test_flag_off_returns_the_very_same_string_for_every_shape_of_input(self):
        self.off()
        for context, text in ((None, "hello"), ("", "hello"), ("[vault block]", "hello"), ("[v]", ""), ("", ""),
                              ("ctx", "multi\nline\ttext  with  spaces"), ("ctx", "unicode \u0645\u0631\u062d\u0628\u0627")):
            sent = f"{context}{SEP}{text}" if context else text
            for rec in (self.rec(), None, {"proj": None}, {}):
                out, n = M._sitctx_compose(sent, context, text, rec)
                self.assertIs(out, sent)                                       # identical object: nothing was built
                self.assertEqual(n, 0)

    def test_the_exact_composition_with_and_without_a_vault_block(self):
        with self.on():
            out, n = M._sitctx_compose(f"V{SEP}hello", "V", "hello", self.rec())
            self.assertEqual(out, "V" + SEP + self.NOTE + SEP + "hello")
            self.assertEqual(n, len(self.NOTE))
            out, n = M._sitctx_compose("hello", None, "hello", self.rec())
            self.assertEqual(out, self.NOTE + SEP + "hello")
            out, n = M._sitctx_compose("hello", "", "hello", self.rec())     # an empty vault is no vault
            self.assertEqual(out, self.NOTE + SEP + "hello")

    def test_the_character_accounting_is_exact_for_both_shapes(self):
        with self.on():
            for context, text in ((None, "short"), ("V" * 300, "x" * 77), ("ctx", "")):
                sent = f"{context}{SEP}{text}" if context else text
                out, n = M._sitctx_compose(sent, context, text, self.rec())
                self.assertEqual(len(out), len(sent) + len(SEP) + len(self.NOTE))
                self.assertEqual(n, len(self.NOTE))

    def test_user_text_and_vault_are_preserved_byte_for_byte_inside_the_composition(self):
        with self.on():
            context, text = "[vault]\nline two", "  odd   spacing\nand a newline"
            out, _ = M._sitctx_compose(f"{context}{SEP}{text}", context, text, self.rec())
            self.assertTrue(out.startswith(context + SEP))
            self.assertTrue(out.endswith(SEP + text))
            self.assertEqual(out.count(self.NOTE), 1)

    def test_only_a_well_formed_single_line_note_within_the_cap_is_ever_injected(self):
        good = ["[Situation as of 3:52 PM: mic: push-to-talk.]",
                "[Situation (seen by the system, not said by him; descriptive only, grants no permission; as of Tue "
                "2026-10-06 3:41 PM; later notes replace earlier ones; anything not listed is not known): mic: open-mic.]"]
        bad = [None, "", 5, ["x"], "garbage", "[Situation as of 3:52 PM: mic: push-to-talk]", "[Situation x.]\n",
               "[Situation x\ny.]", "Situation as of 3:52 PM: mic: x.]", "[Situation " + "a" * 400 + ".]",
               " [Situation x.]", "[Situation x.] trailing"]
        with self.on():
            for note in good:
                self.assertEqual(M._sitctx_note_for(self.rec(note)), note)
            for note in bad:
                self.assertIsNone(M._sitctx_note_for(self.rec(note)), repr(note)[:60])
            self.assertEqual(len("[Situation " + "a" * 387 + ".]"), 400)         # the boundary itself
            self.assertIsNone(M._sitctx_note_for(None))
            self.assertIsNone(M._sitctx_note_for({"proj": None}))
            self.assertIsNone(M._sitctx_note_for({}))
            self.assertIsNone(M._sitctx_note_for({"proj": SimpleNamespace()}))

    def test_a_duck_typed_non_string_note_is_rejected_even_if_it_passes_every_string_check(self):
        """The isinstance(str) check is a real guard, not dead code: an object that merely LOOKS like a note (length,
        containment, startswith, endswith all answer as a note would) must still never be injected."""
        class LooksLikeANote:
            def __len__(self):
                return 20
            def __contains__(self, item):
                return False
            def startswith(self, prefix):
                return True
            def endswith(self, suffix):
                return True
            def __str__(self):
                return "[Situation x.]"
        with self.on():
            self.assertIsNone(M._sitctx_note_for(self.rec(LooksLikeANote())))
            out, n = M._sitctx_compose("hello", None, "hello", self.rec(LooksLikeANote()))
            self.assertEqual((out, n), ("hello", 0))

    def test_the_400_character_boundary_is_inclusive(self):
        at_cap = "[Situation " + "a" * 387 + ".]"
        over = "[Situation " + "a" * 388 + ".]"
        self.assertEqual((len(at_cap), len(over)), (400, 401))
        with self.on():
            self.assertEqual(M._sitctx_note_for(self.rec(at_cap)), at_cap)
            self.assertIsNone(M._sitctx_note_for(self.rec(over)))

    def test_flag_off_never_yields_a_note_even_for_a_perfect_one(self):
        self.off()
        self.assertIsNone(M._sitctx_note_for(self.rec()))

    def test_the_marker_is_fixed_metadata_only(self):
        lines = []
        with mock.patch.object(M, "log", lines.append), mock.patch.dict(M.CFG, {M._SITCTX_FLAG: True}):
            M._sitctx_marker("BASE-secret hunter2 /Users/yousif", "FINAL-secret sk-live-abc rm -rf", 44)
        self.assertEqual(lines, ["[sitinj] flag=on injected=yes note_chars=44 base_chars=%d final_chars=%d"
                                 % (len("BASE-secret hunter2 /Users/yousif"), len("FINAL-secret sk-live-abc rm -rf"))])
        self.assertRegex(lines[0], MARKER)
        for bad in ("secret", "hunter2", "/Users", "sk-live", "rm -rf"):
            self.assertNotIn(bad, lines[0])

    def test_the_marker_for_off_and_a_failing_log(self):
        lines = []
        with mock.patch.object(M, "log", lines.append), mock.patch.dict(M.CFG):
            M.CFG.pop(M._SITCTX_FLAG, None)
            M._sitctx_marker("abc", "abc", 0)
        self.assertEqual(lines, ["[sitinj] flag=off injected=no note_chars=0 base_chars=3 final_chars=3"])

        def bad(_l):
            raise OSError("disk full")
        with mock.patch.object(M, "log", bad):
            M._sitctx_marker("abc", "abc", 0)                                 # never raises


class InjectionOffFlowTests(ShadowBase):
    """Flag OFF (the key absent): the model input is byte-for-byte what it was before 3C."""

    async def test_the_default_is_off_and_the_model_input_is_exactly_the_text(self):
        self.assertNotIn(M._SITCTX_FLAG, M.CFG)
        await self.reply("what is two plus two")
        await self.reply("and again")
        self.assertEqual(self.model_inputs(), ["what is two plus two", "and again"])
        self.assertGreater(len(self.notes), 0)                                  # a projection exists, and is not used

    async def test_a_vault_context_is_unchanged_with_the_flag_off(self):
        await self.reply_ctx("keep me", VAULT_BLOCK)
        self.assertEqual(self.model_inputs(), [f"{VAULT_BLOCK}{SEP}keep me"])

    async def test_no_extra_separator_or_character_is_added_with_the_flag_off(self):
        for text in ("a", "two  spaces", "trailing newline\n", "  leading", "x" * 500):
            await self.reply(text)
        self.assertEqual(self.model_inputs(), ["a", "two  spaces", "trailing newline\n", "  leading", "x" * 500])

    async def test_the_marker_reports_off_and_equal_character_counts(self):
        await self.reply("hello there")
        self.assertEqual(self.markers(), ["[sitinj] flag=off injected=no note_chars=0 base_chars=11 final_chars=11"])

    async def test_explicit_false_and_look_alike_values_are_also_off(self):
        for value in (False, None, "true", 1, "yes"):
            M.CFG[M._SITCTX_FLAG] = value
            await self.reply("probe")
            self.assertEqual(self.model_inputs()[-1], "probe", repr(value))

    async def test_attribution_and_adoption_are_unchanged_with_the_flag_off(self):
        await self.reply("one")
        self.mic["mode"] = "ptt"
        await self.reply("two")
        await self.reply("three")
        self.assertEqual([re.search(r"projection=(\w+)", ln).group(1) for ln in self.sitctx()], ["baseline", "delta", "none"])
        self.assertTrue(all("delivery=claimed" in ln for ln in self.sitctx()))
        await self.send_unclaimed("four")
        self.assertIn("delivery=unclaimed", self.sitctx()[-1])


class InjectionOnFlowTests(ShadowBase):
    """Flag ON: the note joins the model input at the send point, and only there."""

    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.flag_on()

    def last_input(self):
        return self.model_inputs()[-1]

    async def test_the_baseline_note_is_injected_with_the_exact_composition(self):
        await self.reply("what is two plus two")
        note = self.notes[-1]
        self.assertTrue(note.startswith("[Situation (seen by the system"))
        self.assertEqual(self.model_inputs(), [note + SEP + "what is two plus two"])
        self.assertRegex(self.sitctx()[-1], r"projection=baseline fields=mic,background length=%d delivery=claimed" % len(note))

    async def test_a_vault_block_stays_first_and_the_note_follows_it(self):
        await self.reply_ctx("a question", VAULT_BLOCK)
        note = self.notes[-1]
        self.assertEqual(self.last_input(), VAULT_BLOCK + SEP + note + SEP + "a question")

    async def test_an_unchanged_state_produces_no_duplicate_note(self):
        await self.reply("one")
        await self.reply("two")
        await self.reply("three")
        self.assertEqual(len(self.notes), 1)
        self.assertEqual(self.model_inputs()[1:], ["two", "three"])             # byte-identical to flag OFF from here on

    async def test_a_delta_note_is_injected_when_a_field_changes(self):
        await self.reply("one")
        self.mic["mode"] = "ptt"
        await self.reply("two")
        self.assertEqual(len(self.notes), 2)
        self.assertIn("mic: push-to-talk", self.notes[-1])
        self.assertNotIn("background", self.notes[-1])
        self.assertEqual(self.last_input(), self.notes[-1] + SEP + "two")
        self.assertRegex(self.sitctx()[-1], r"projection=delta fields=mic")

    async def test_a_retraction_is_injected_by_the_existing_projector_rules(self):
        await self.reply("one")
        def boom():
            raise RuntimeError("ledger unreadable")
        self.brain.live_background = boom                                        # the background field becomes unavailable
        await self.reply("two")
        self.assertEqual(self.notes[-1].split(": ", 1)[-1], "background agents running: no longer known.]")
        self.assertEqual(self.last_input(), self.notes[-1] + SEP + "two")
        self.assertRegex(self.sitctx()[-1], r"projection=retraction fields=background")
        await self.reply("three")
        self.assertEqual(self.last_input(), "three")                             # retracted exactly once

    async def test_a_clear_gives_a_fresh_baseline_in_the_model_input(self):
        await self.reply("one")
        await self.reply("two")
        M._sitctx_invalidate()
        await self.reply("three")
        self.assertEqual(self.last_input(), self.notes[-1] + SEP + "three")
        self.assertEqual(len(self.notes), 2)
        self.assertTrue(self.notes[-1].startswith("[Situation (seen by the system"))   # the full header again
        await self.reply("four")
        self.assertEqual(self.last_input(), "four")

    async def test_a_compact_gives_a_fresh_baseline_in_the_model_input(self):
        await self.reply("one")
        M._sitctx_invalidate()                                                   # a compact bumps the same counter, same session
        self.assertEqual((M._SITCTX_GEN["n"], self.brain.session_id), (1, F.SID))
        await self.reply("two")
        self.assertEqual(self.last_input(), self.notes[-1] + SEP + "two")
        self.assertEqual(len(self.notes), 2)

    async def test_a_changed_session_gives_a_fresh_baseline_in_the_model_input(self):
        await self.reply("one")
        with mock.patch.object(F, "SID", "sess-2"):
            self.set_script(lambda c, m: c.feed(F.init(), F.asst("Cleared."), F.echo(m["uuid"], "<command-name>/clear"),
                                                F.result("Cleared.", num_turns=0)))
            await self.brain.command("/clear")
            self.set_script(lambda c, m: c.feed(*F.fg_run(m["uuid"], "Hello there.")))
            self.assertEqual(M._SITCTX_GEN["n"], 0)
            await self.reply("two")
            await self.reply("three")
        self.assertEqual(self.model_inputs()[-2], self.notes[-1] + SEP + "two")
        self.assertEqual(self.model_inputs()[-1], "three")
        self.assertEqual(len(self.notes), 2)

    async def test_an_epoch_change_gives_a_fresh_baseline_in_the_model_input(self):
        await self.reply("one")
        await self.brain.rebuild("desync")
        await self.reply("two")
        self.assertEqual(self.last_input(), self.notes[-1] + SEP + "two")
        self.assertEqual(len(self.notes), 2)

    async def test_an_unclaimed_send_does_not_adopt_so_the_next_send_carries_the_note_again(self):
        await self.send_unclaimed("first")
        self.assertIsNone(M._SITCTX["state"])
        self.assertEqual(self.model_inputs()[-1], self.notes[-1] + SEP + "first")  # it WAS in the abandoned message
        await self.reply("second")
        self.assertEqual(self.last_input(), self.notes[-1] + SEP + "second")
        self.assertEqual(len(self.notes), 2)
        await self.reply("third")
        self.assertEqual(self.last_input(), "third")

    async def test_an_identity_mismatch_does_not_adopt_state(self):
        await self.reply("one")
        adopted = M._SITCTX["state"]
        self.mic["mode"] = "ptt"
        real_begin = M._sitctx_begin

        def begin_then_lose_the_stream(brain):
            rec = real_begin(brain)
            brain._need_rebuild = "stream_lost"
            return rec
        with mock.patch.object(M, "_sitctx_begin", begin_then_lose_the_stream):
            await self.reply("two")
        self.assertEqual(self.model_inputs()[-1], self.notes[-1] + SEP + "two")   # injected, because it was composed first
        self.assertIn("delivery=unattributed", self.sitctx()[-1])
        self.assertIs(M._SITCTX["state"], adopted)                                  # but never adopted
        await self.reply("three")
        self.assertEqual(self.last_input(), self.notes[-1] + SEP + "three")        # re-baselined after the rebuild

    async def test_projection_failure_injects_nothing(self):
        def boom(*a, **k):
            raise RuntimeError("secret detail /Users/yousif hunter2")
        with mock.patch.object(C, "project", boom):
            await self.reply("hello")
        self.assertEqual(self.model_inputs(), ["hello"])
        self.assertEqual(self.markers(), ["[sitinj] flag=on injected=no note_chars=0 base_chars=5 final_chars=5"])
        self.assertEqual(self.mouth.chunks, ["Hello there."])                     # the turn itself was normal
        self.assertIn("[sitctx] projection_failure category=projection", self.lines)

    async def test_a_snapshot_failure_injects_nothing(self):
        self.reader.snapshot = mock.Mock(side_effect=RuntimeError("boom"))
        await self.reply("hello")
        self.assertEqual(self.model_inputs(), ["hello"])

    async def test_a_malformed_projection_is_never_injected(self):
        bad = [None, "", "garbage", "[Situation x", "[Situation a\nb.]", "[Situation " + "a" * 400 + ".]", 12345]
        for note in bad:
            with mock.patch.object(C, "project", lambda *a, _n=note, **k: C.Projection(_n, C.ProjectionState())):
                await self.reply("probe")
            self.assertEqual(self.model_inputs()[-1], "probe", repr(note)[:40])

    async def test_without_a_shadow_reader_nothing_is_injected_even_with_the_flag_on(self):
        with mock.patch.object(M, "_SITCTX_READER", None):
            await self.reply("hello")
        self.assertEqual(self.model_inputs(), ["hello"])
        self.assertEqual(self.markers(), ["[sitinj] flag=on injected=no note_chars=0 base_chars=5 final_chars=5"])   # honest audit

    async def test_the_character_accounting_is_computed_from_the_exact_strings(self):
        await self.reply("plain question")
        await self.reply_ctx("with vault", VAULT_BLOCK)
        self.mic["mode"] = "ptt"
        await self.reply("after a delta")
        bases = ["plain question", f"{VAULT_BLOCK}{SEP}with vault", "after a delta"]
        injected = [self.notes[0], None, self.notes[1]]          # the second send was already told, so no note
        finals = self.model_inputs()
        self.assertEqual(len(self.notes), 2)
        expected_markers = []
        for base, note, final in zip(bases, injected, finals):
            if note:
                self.assertEqual(len(final), len(base) + len(SEP) + len(note))
            else:
                self.assertEqual(final, base)
            expected_markers.append(marker_for("on", note, base, final))
        self.assertEqual(self.markers(), expected_markers)

    async def test_the_marker_is_logged_before_the_send_and_matches_the_regex(self):
        await self.reply("hello")
        self.assertEqual(self.order[:2], ["marker", "query"])
        for ln in self.markers():
            self.assertRegex(ln, MARKER)
        self.assertEqual(self.markers(), [marker_for("on", self.notes[0], "hello", self.model_inputs()[0])])

    async def test_the_note_is_never_spoken(self):
        for text in ("one", "two"):
            await self.reply(text)
        self.mic["mode"] = "ptt"
        await self.reply("three")
        spoken = " ".join(self.mouth.chunks + self.mouth.said)
        for note in self.notes:
            self.assertNotIn(note, spoken)
        self.assertNotIn("Situation", spoken)
        self.assertEqual(self.mouth.chunks, ["Hello there."] * 3)

    async def test_the_note_never_reaches_telemetry_or_the_audit_lines(self):
        await self.reply("one")
        self.mic["mode"] = "ptt"
        await self.reply("two")
        blob = "\n".join(self.lines)
        for note in self.notes:
            self.assertNotIn(note, blob)
        self.assertEqual(len(re.findall(r"\[Situation", blob)), 0)
        for ln in self.lines:
            if ln.startswith("[sit] "):
                self.assertNotIn("projection", ln)
                self.assertNotIn("inject", ln)
        for ln in self.markers():
            self.assertRegex(ln, MARKER)
        for ln in self.sitctx():
            self.assertRegex(ln, LINE)                                            # the 3B format is unchanged

    async def test_the_note_is_not_in_the_ledger_beyond_the_preview_and_never_with_a_vault_block(self):
        await self.reply_ctx("with vault", VAULT_BLOCK)
        note = self.notes[-1]
        self.assertGreater(len(VAULT_BLOCK), 200)
        for row in list(self.led.turns.values()) + list(self.led.actions.values()) + list(self.led.tasks.values()):
            self.assertNotIn("Situation", str(row))                               # a long vault block fills the 200-char preview
        blob = open(os.path.join(self.tmp.name, "ledger.jsonl"), encoding="utf-8").read()
        self.assertNotIn(note, blob)

    async def test_the_ledger_intent_is_exactly_the_preview_of_the_model_input(self):
        """DOCUMENTED, not hidden: ledger intent = first 200 characters of what ask_stream received. Without a
        vault block that is the start of the note itself (its fixed header, never an id or a path). Keeping it out
        of the preview would need brain/router to pass an explicit intent, which is outside this change."""
        await self.reply("what is two plus two")
        handle = next(h for h in self.brain.router.handles.values() if h.kind == "foreground")
        final = self.model_inputs()[-1]
        self.assertEqual(handle.intent, L.redact(final))
        self.assertNotIn("what is two plus two", handle.intent)                    # the user's words fall out of the preview

    async def test_last_turn_and_the_turn_text_never_see_the_note(self):
        tree = ast.parse(inspect.getsource(M))
        amain = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "amain")
        handle = next(n for n in ast.walk(amain) if isinstance(n, ast.AsyncFunctionDef) and n.name == "handle")
        names = {n.id for n in ast.walk(handle) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(handle) if isinstance(n, ast.Attribute)}
        for forbidden in ("_sitctx_compose", "_sitctx_note_for", "_sitctx_marker", "_sitctx_inject_enabled", "_SITCTX_FLAG",
                          "situation_context_mod", "_SITCTX_READER", "_SITCTX"):
            self.assertNotIn(forbidden, names, forbidden)

    async def test_the_voice_and_the_gates_see_nothing_new(self):
        await self.reply("one")
        for forbidden in ("permission", "classify", "tier"):
            self.assertFalse([ln for ln in self.lines if forbidden in ln.lower() and "[sitinj]" in ln])
        self.assertEqual(self.mouth.chunks, ["Hello there."])


class InjectionStructureTests(unittest.TestCase):
    def speak_reply_tree(self):
        return ast.parse(inspect.getsource(M.speak_reply).replace("\n    ", "\n", 0))

    def test_ask_stream_receives_the_composed_input_and_end_receives_the_same_string(self):
        tree = self.speak_reply_tree()
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "ask_stream"]
        self.assertEqual(len(calls), 1)
        self.assertIsInstance(calls[0].args[0], ast.Name)
        # Memory 2.0 injection (default OFF) adds ONE stage after the composition: send_input is model_input itself,
        # the very same string, unless the memory note is injected. Its only definition is _mem2_compose(model_input, ...).
        self.assertEqual(calls[0].args[0].id, "send_input")
        defs = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Tuple) and any(isinstance(e, ast.Name) and e.id == "send_input" for e in t.elts) for t in n.targets)]
        calls_ = [d.value.value for d in defs]                                  # each is `await _mem2_compose(...)`
        self.assertTrue(all(isinstance(d.value, ast.Await) for d in defs))
        self.assertEqual([ast.unparse(c.args[0]) for c in calls_], ["model_input"])
        self.assertEqual([getattr(c.func, "id", None) for c in calls_], ["_mem2_compose"])
        ends = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "_sitctx_end"]
        self.assertEqual([ast.unparse(e.args[2]) for e in ends], ["send_input"])   # the attribution cross-check sees the real input

    def test_model_input_has_exactly_one_definition_and_it_is_the_compose_helper(self):
        tree = self.speak_reply_tree()
        defs = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Tuple) and any(isinstance(e, ast.Name) and e.id == "model_input" for e in t.elts) for t in n.targets)]
        self.assertEqual(len(defs), 1)
        self.assertEqual(getattr(defs[0].value.func, "id", None), "_sitctx_compose")
        loose = [n for n in ast.walk(tree) if isinstance(n, (ast.Assign, ast.AugAssign))
                 and any(isinstance(t, ast.Name) and t.id == "model_input" for t in (n.targets if isinstance(n, ast.Assign) else [n.target]))]
        self.assertEqual(loose, [])

    def test_the_marker_call_comes_before_the_try_that_sends(self):
        tree = self.speak_reply_tree()
        marker = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "_sitctx_marker"]
        sends = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "ask_stream"]
        self.assertEqual((len(marker), len(sends)), (1, 1))
        self.assertLess(marker[0].lineno, sends[0].lineno)

    def test_the_new_helpers_are_called_only_from_speak_reply_and_each_other(self):
        tree = ast.parse(inspect.getsource(M))
        callers = {}
        for fn in ast.walk(tree):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for n in ast.walk(fn):
                    if isinstance(n, ast.Call) and getattr(n.func, "id", None) in ("_sitctx_compose", "_sitctx_note_for",
                                                                                  "_sitctx_marker", "_sitctx_inject_enabled"):
                        callers.setdefault(n.func.id, set()).add(fn.name)
        self.assertEqual(callers["_sitctx_compose"], {"speak_reply"})
        self.assertEqual(callers["_sitctx_marker"], {"speak_reply"})
        self.assertEqual(callers["_sitctx_note_for"], {"_sitctx_compose"})
        self.assertEqual(callers["_sitctx_inject_enabled"], {"_sitctx_note_for", "_sitctx_marker"})

    def test_cfg_is_read_only_by_the_flag_helper_and_the_flag_key_appears_once(self):
        src = inspect.getsource(M)
        self.assertEqual(len(re.findall(r'["\']situation_context_inject["\']', src)), 1)   # the one key literal: the constant
        tree = ast.parse(src)
        readers = set()
        for fn in ast.walk(tree):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and fn.name.startswith("_sitctx"):
                if any(isinstance(n, ast.Name) and n.id == "CFG" for n in ast.walk(fn)):
                    readers.add(fn.name)
        self.assertEqual(readers, {"_sitctx_inject_enabled"})                       # no other shadow code reads the config

    def test_there_is_still_no_second_flag_or_authority_coupling(self):
        src = inspect.getsource(M)
        for word in ("situation_inject", "inject_situation", "SITCTX_INJECT", "situation_enable", "situation_on"):
            self.assertNotIn(word, src)
        for name in ("addressee.py", "router.py", "brain.py", "ledger.py", "speech_policy.py", "config.py",
                     "situation.py", "verify.py", "evidence.py"):
            with open(os.path.join(PKG, name), encoding="utf-8") as f:
                text = f.read()
            self.assertNotIn("situation_context_inject", text, name)
            self.assertNotIn("_sitctx", text, name)


if __name__ == "__main__":
    unittest.main()
