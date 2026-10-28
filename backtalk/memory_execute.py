"""Memory 2.0, Phase 8.2d: EXECUTE. The one flow that turns a reviewed decision into a stored memory, and the typed-word
reconcile. A LIBRARY: nothing in Jarvis calls it, and it is exercised only against temporary stores. The terminal, the
clock, the nonce source and the plan builder are all injected, so nothing here reads a file, a clock or the keyboard on its
own.

    commit(...)      startup check -> plan -> preview and typed challenge -> FRESH re-plan and compare -> receipts
                     (first use) -> intent through `ReceiptLog.append_intent` ONLY -> store install -> snapshot -> proof
                     -> done / aborted / anomaly
    reconcile(...)   the typed word appends the ONE record `audit` proposed, and nothing else

THE RULES THIS ENFORCES (each is a refusal with a code, never a repair):
  * Nothing starts unless `RC.assess` says proceed (first use, or a clean log). An open intent, a blocked log, a store that
    does not match the log: refused before a plan is built.
  * The person confirms a Plan. Immediately before any receipt or store call the plan is REBUILT from scratch and compared
    with `compare_plans`; any difference (including `clock_regressed`) aborts with no receipt and no store call.
  * An intent is appended ONLY through `ReceiptLog.append_intent`, which takes its own fresh snapshot and runs the live
    check inside the receipts lock. There is no other intent path in this module.
  * The store is called only after the intent is durable, with an Authorization the store itself verifies (item digest,
    live state digest, operation, its own clock window, the single-use nonce). A Plan, token or answer is never itself an
    authorization.
  * After the install call (whether it returned or refused) the store is snapshotted and `classify_open_intent` decides what
    the log is told: COMMITTED -> `done`, NOT_WRITTEN -> `aborted`, anything else -> `anomaly`. Never from the exception.
  * A store that requires authorization (the real vault) is refused unless a second confirmation (`tier2`, injected) is
    passed: it is prepared before anything is shown, checked against the plan, and asked AFTER the typed answer, bound to
    that plan and nonce, and its evidence is durable BEFORE the intent. Its refusals write no receipt. It is possession,
    not identity authentication, and the store cannot verify it. `reconcile` never asks for it.
  * A crash or interrupt during the install is not classified in-process: the open intent stays, and the next start shows it
    (`reconcile_required`), exactly as the receipt protocol says.

BUDGETS. `PREVIEW_BUDGET_S` (45 s) was measured by W2 (worst 3.30 s) and kept. `E3_DEADLINE_S` (90 s, typed answer to the
store call) and `REST_S` (30 s, durable intent to the store call) are PROVISIONAL and were NOT measured by W2; they are
constants here, not claims.

LIMITS, stated and pinned by tests: the receipts lock does not exclude a non-cooperating store writer or a direct file
edit, and the interval between the intent's snapshot and its write is not atomic against them; a stale intent can never
authorize a write (the store's own state check and the post-install proof remain the barriers). The typed challenge is an
accidental-error confirmation, not authentication; same-account code can compute it. The terminal rule (a real terminal on
both ends) is the CLI's, and it is a convention for this program, not a defense against code running as you.
"""
import re
from dataclasses import dataclass

from backtalk import memory as M
from backtalk import memory_commit as C
from backtalk import memory_gate as G
from backtalk import memory_receipts as RC
from backtalk import memory_store as S

PREVIEW_BUDGET_S = 45        # W2-measured (worst 3.30 s) and kept
E3_DEADLINE_S = 90           # PROVISIONAL, not measured by W2: from the typed answer to just before the store call
REST_S = 30                  # PROVISIONAL, not measured by W2: from the durable intent to the store call
RECONCILE_WORD = "reconcile"
_HEX32 = re.compile(r"^[0-9a-f]{32}\Z")


@dataclass(frozen=True)
class Outcome:
    status: str                    # "committed" | "declined" | "refused" | "aborted" | "anomaly" | "unclassified" | "reconciled"
    code: str                      # a stable reason
    memory_id: object = None
    intent_seq: object = None
    details: tuple = ()


def _stamp(clock):
    value = clock()
    if not (isinstance(value, str) and M.parse_ts(value) is not None):
        raise RC.ReceiptError("bad_clock")
    return value


def _refuse(code, details=()):
    return Outcome("refused", code, details=tuple(details))


def commit(*, make_plan, store, receipts_dir, ask, say, clock, monotonic, nonce_fn, tier2=None) -> Outcome:
    """Run one commit. `make_plan()` must rebuild a Plan from scratch on every call (fresh ledger, pack bytes, sources,
    store read and trusted clock); it raises `CommitError` on any refusal. `tier2` is the injected second confirmation
    (`prepare`, `check_plan`, `confirm`); a store that requires authorization (the real vault) is refused without it."""
    if store.require_authorization and tier2 is None:
        return _refuse("tier2_required")
    # 1. startup check (read-only). Nothing is planned, shown or written unless it says proceed.
    start = RC.assess(receipts_dir, store)
    if not start.proceed:
        return _refuse(start.code, (f"row_{start.row}",) + tuple(str(d) for d in start.details[:4]))
    if tier2 is not None:                              # configuration, before anything sensitive is read or shown
        code = tier2.prepare()
        if code:
            return _refuse("tier2:" + code)
    # 2. plan, preview, typed challenge
    try:
        first = make_plan()
        preview = C.render_preview(first)
    except C.CommitError as e:
        return _refuse("plan_refused:" + e.code, e.details)
    if tier2 is not None:                              # kind, approval and limits, before the person is asked anything
        code = tier2.check_plan(first)
        if code:
            return _refuse("tier2:" + code)
    nonce = nonce_fn()
    if not (isinstance(nonce, str) and _HEX32.match(nonce)):
        return _refuse("bad_nonce")
    for line in preview.bound:
        say(line)
    say(C.NOTICE)
    for line in preview.info:
        say(line)
    try:
        issued_at = _stamp(clock)
    except RC.ReceiptError:
        return _refuse("bad_clock")                    # nothing has been written yet
    typed = ask(f"To commit, type exactly:  {C.expected_answer(first, nonce)}   > ")
    try:
        answered_at = _stamp(clock)
    except RC.ReceiptError:
        return _refuse("bad_clock")
    t_answer = monotonic()
    try:
        confirmed_at = C.check_answer(typed, first, nonce, issued_at, answered_at)
    except C.CommitError as e:
        return Outcome("declined", e.code, details=e.details)
    if tier2 is not None:                              # bound to plan #1 and this nonce; recorded before any intent
        second = tier2.confirm(first, nonce, preview.bound)
        if not second.ok:
            return _refuse("tier2:" + str(second.code))
        try:
            confirmed_at = _stamp(clock)               # the confirmation instant; E3 runs from this last human act
        except RC.ReceiptError:
            return _refuse("bad_clock")
        t_answer = monotonic()
    # 3. FRESH re-plan, compared with what the person confirmed, before any receipt or store call
    try:
        plan = make_plan()
    except C.CommitError as e:
        return _refuse("replan_refused:" + e.code, e.details)
    changed = C.compare_plans(first, plan)
    if changed:
        return _refuse("plan_changed", changed)
    if M.parse_ts(plan.planned_at) < M.parse_ts(confirmed_at):
        return _refuse("clock_regressed")
    if monotonic() - t_answer > E3_DEADLINE_S:
        return _refuse("deadline_exceeded")
    # 4. receipts: re-check immediately before writing; create only on first use (an existing store is never baselined)
    log = RC.ReceiptLog(receipts_dir)
    again = RC.assess(receipts_dir, store)
    if not again.proceed:
        return _refuse(again.code, (f"row_{again.row}",))
    try:
        if again.first_use:
            log.create(again.snapshot, _stamp(clock))
        # 5. the intent, through the only door
        apply_now = _stamp(clock)
        fields = RC.intent_fields(store.snapshot(), plan, nonce=nonce, confirmed_at=confirmed_at, apply_now=apply_now)
        intent = log.append_intent(fields, _stamp(clock), store)
    except RC.ReceiptError as e:
        return _refuse("receipts:" + e.reason, e.details)
    t_intent = monotonic()
    # 6. the store call, only now, with an Authorization the store verifies itself
    install_error = None
    if monotonic() - t_answer > E3_DEADLINE_S or t_intent - t_answer > E3_DEADLINE_S:
        install_error = ("deadline_exceeded",)
    elif monotonic() - t_intent > REST_S:
        install_error = ("rest_exceeded",)
    else:
        auth = S.Authorization(plan.operation, plan.item_digest, plan.state_digest, nonce, confirmed_at,
                               plan.context_digest, "typed-terminal")
        try:
            if plan.operation == "put":
                store.put(plan.item, plan.evidence, now=apply_now, authorization=auth)
            else:
                store.apply(plan.decision, G.Proposal(plan.item, plan.evidence), apply_now, authorization=auth)
        except S.StoreError as e:                      # a refusal: the store decided, and said why (a code)
            install_error = (e.reason,) + tuple(str(d) for d in e.details[:3])
    # 7. settle from the store's actual state, never from the exception
    return _settle(log, store, intent, install_error, clock)


def _settle(log, store, intent, install_error, clock) -> Outcome:
    snap = store.snapshot()
    if not snap.ok:
        return Outcome("unclassified", "snapshot:" + snap.reason, intent["memory_id"], intent["seq"],
                       tuple(install_error or ()))
    proof = RC.classify_open_intent(intent, snap)
    try:
        if proof.kind == "COMMITTED":
            log.append(RC.done_fields(intent["seq"], snap, recovered=install_error is not None), _stamp(clock))
            return Outcome("committed", "committed", intent["memory_id"], intent["seq"], tuple(install_error or ()))
        if proof.kind == "NOT_WRITTEN":
            log.append(RC.aborted_fields(intent["seq"], "crashed_before_write", snap.state_digest), _stamp(clock))
            return Outcome("aborted", "not_written", intent["memory_id"], intent["seq"], tuple(install_error or ()))
        log.append(RC.anomaly_fields(intent["seq"], proof.codes, snap.state_digest), _stamp(clock))
        return Outcome("anomaly", "inconsistent", intent["memory_id"], intent["seq"], proof.codes)
    except RC.ReceiptError as e:                       # the log could not be told; the open intent remains for reconcile
        return Outcome("unclassified", "receipts:" + e.reason, intent["memory_id"], intent["seq"], e.details)


def reconcile(*, store, receipts_dir, ask, say, clock) -> Outcome:
    """The typed word `reconcile` appends exactly the one record `audit` proposes for exactly one open intent. It cannot
    change the classification, and nothing else is overridable: a blocked log, a corrupt chain, a snapshot that is not ok,
    a marker, a state gap or no open intent is refused with its code and nothing is written."""
    a = RC.assess(receipts_dir, store)
    if a.row != 7 or a.proposed is None:
        return _refuse(a.code, (f"row_{a.row}",) + tuple(str(d) for d in a.details[:4]))
    for line in a.facts + a.inferences:
        say(line)
    say("the one record that would be appended: " + a.proposed["action"] +
        (" " + ", ".join(a.proposed["codes"]) if a.proposed["action"] == "anomaly" else ""))
    if ask(f"Type the word  {RECONCILE_WORD}  to append exactly that record: ") != RECONCILE_WORD:
        return Outcome("declined", "not_confirmed")
    b = RC.assess(receipts_dir, store)                  # nothing may have moved since it was shown
    if b.row != 7 or b.proposed != a.proposed:
        return _refuse("state_changed", (f"row_{b.row}",))
    log = RC.ReceiptLog(receipts_dir)
    try:
        rec = log.append(b.proposed, _stamp(clock), expect_prev=b.state.last_hash)
    except RC.ReceiptError as e:
        return _refuse("receipts:" + e.reason, e.details)
    return Outcome("reconciled", rec["action"], b.state.open_intent["memory_id"], rec["intent_seq"])
