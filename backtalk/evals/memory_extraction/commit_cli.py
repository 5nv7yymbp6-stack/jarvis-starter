"""The Memory 2.0 commit program (Phase 8.2d): you, at a terminal, committing ONE reviewed decision to a Memory 2.0 store.

    python -m backtalk.evals.memory_extraction.commit_cli --status               # the startup check, no terminal needed
    python -m backtalk.evals.memory_extraction.commit_cli --audit                # facts, inferences, the proposed record
    python -m backtalk.evals.memory_extraction.commit_cli --commit CANDIDATE     # needs a real terminal
    python -m backtalk.evals.memory_extraction.commit_cli --reconcile            # needs a real terminal

TEMPORARY STORES BY DEFAULT. A store under the real vault is refused by the store itself (`real_vault_refused`) unless
`--real-vault` is given, which is accepted only for exactly the real vault path and never with a test clock. On the real
vault every write still needs: a real terminal, the typed challenge, the Tier 2 phone confirmation (Keychain bot), a
user-created pilot approval file, a locked candidate in the review ledger, and the store's own Authorization check.
Nothing here restarts, imports into or is imported by Jarvis.

RULES (enforced in code and tests):
  * Committing and reconciling need a TERMINAL on both input and output. A pipe, a file or a script is refused before
    anything is read. Voice and chat are never a confirmation channel.
  * Every commit goes through `memory_execute.commit`: startup check, plan, preview and typed challenge, a FRESH re-plan
    compared with what you confirmed, receipts, the intent through `ReceiptLog.append_intent` only, the store install, the
    proof, and the receipt that records it. This program has no other path to the store or to the receipts.
  * `--status` and `--audit` are read-only.
  * It writes the receipts (`commits.*`, in the review root) and the Memory 2.0 store of the vault it was pointed at, only
    through the library above, and never the review ledger, the pack, the index or the dreaming system.
"""
import argparse
import json
import os
import secrets
import sys
import time
from datetime import datetime, timezone
from types import SimpleNamespace

from backtalk import memory_commit as C
from backtalk import memory_evidence as EV
from backtalk import memory_execute as EX
from backtalk import memory_extract as X
from backtalk import memory_receipts as RC
from backtalk import memory_review as R
from backtalk import memory_store as S
from backtalk import memory_tier2 as T
from backtalk.evals.memory_extraction import evaluate as E
from backtalk.evals.memory_extraction import review_cli as RV
from backtalk.evals.memory_extraction import scan

NO_TTY = "Committing needs a real terminal for both input and output. Nothing was read or written."
REAL_VAULT_MISUSE = ("--real-vault applies only to the real vault itself and never with a test clock. "
                     "Nothing was read or written.")
REAL_VAULT_REFUSED = ("This phase commits to temporary stores only: that vault is the real vault and the store refused it. "
                      "Nothing was read or written.")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_private(path) -> dict:
    """{ledger seq: {"value", "evidence"}} from the private side file, or {} when there is none."""
    out = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    raise R.ReviewError("ledger_corrupt", ("unreadable_private_line",)) from None
                if isinstance(rec, dict) and isinstance(rec.get("seq"), int):
                    out[rec["seq"]] = {"value": rec.get("value"), "evidence": rec.get("evidence")}
    return out


def production_provenance(transcripts, vault):
    """The source check: re-open the source and report what no longer matches (an empty list is "still says it")."""
    return lambda view: E.verify_provenance(SimpleNamespace(**view), transcripts, vault)


def production_rederive(transcripts, vault, generated_at):
    """Re-derive the candidate from scratch with the same pipeline that built the pack (what W2 measured). None if the
    pipeline no longer produces it."""
    def rederive(view):
        cands, conflicts, meta, _ = scan.run_pipeline(transcripts, vault, generated_at)
        for cand in X.review_pack(cands, conflicts, meta)["candidates"]:
            if cand["id"] == view["id"]:
                return cand
        return None
    return rederive


def make_plan_builder(root, run_dir, candidate, store, vault, transcripts, clock, monotonic):
    """A callable that rebuilds the Plan from scratch each time it is called: pack bytes, ledger, private side file,
    sources and the store are all read again."""
    def build():
        failed = None
        try:
            raw, pack_sha, _run = RV.read_pack(run_dir)
            records = RV.read_ledger(os.path.join(root, RV.LEDGER))
            private = read_private(os.path.join(root, RV.PRIVATE))
            pack = R.Pack(raw)
            generated_at = raw["meta"]["generated_at"]
        except (R.ReviewError, OSError, ValueError, KeyError, TypeError) as e:
            failed = type(e).__name__                    # a class name only: never text that could carry a value
        if failed is not None:
            raise C.CommitError("inputs_unreadable", (failed,))
        return C.plan(pack=pack, pack_sha=pack_sha, records=records, private=private or None, candidate=candidate,
                      store=store, provenance=production_provenance(transcripts, vault),
                      rederive=production_rederive(transcripts, vault, generated_at),
                      clock=clock, monotonic=monotonic, preview_budget_s=EX.PREVIEW_BUDGET_S)
    return build


def receipt_reader(root):
    """The receipt records for the Tier 2 gate: [] when there is no log yet, EvidenceError when it cannot be read."""
    def records():
        log = RC.ReceiptLog(root)
        try:
            return log.records() if os.path.exists(log.path) else []
        except RC.ReceiptError:
            raise EV.EvidenceError("receipts_unreadable") from None
    return records


def build_tier2(store, root, clock, monotonic, confirmer_factory):
    """None for a store that does not require authorization (a temporary store never builds or calls a confirmer).
    For one that does, a gate whose confirmer is None when it cannot be configured, so the commit refuses closed.
    `confirmer_factory(monotonic)` is the only place the Keychain and the network can be reached."""
    if not store.require_authorization:
        return None
    try:
        confirmer = confirmer_factory(monotonic)
    except Exception:
        confirmer = None
    return EV.Tier2Gate(confirmer, root, clock, receipt_reader(root))


def main(argv=None, *, ask=None, say=None, tty=None, clock=None, monotonic=None, nonce_fn=None, store_clock=None,
         confirmer_factory=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=scan.DEFAULT_OUT)
    ap.add_argument("--run")
    ap.add_argument("--vault", default=scan.DEFAULT_VAULT)
    ap.add_argument("--transcripts", default=scan.DEFAULT_TRANSCRIPTS)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--audit", action="store_true")
    mode.add_argument("--commit", metavar="CANDIDATE")
    mode.add_argument("--reconcile", action="store_true")
    ap.add_argument("--real-vault", action="store_true")
    a = ap.parse_args(argv)
    say = say or print
    ask = ask or input
    clock = clock or _now
    monotonic = monotonic or time.monotonic
    nonce_fn = nonce_fn or (lambda: secrets.token_hex(16))
    confirmer_factory = confirmer_factory or T.configured_confirmer
    writing = bool(a.commit) or a.reconcile
    if writing:
        if tty is None:                       # (`tty` is a test hook; the command line has no way to set it)
            tty = sys.stdin.isatty() and sys.stdout.isatty()
        if not tty:
            say(NO_TTY)
            return 2
    if a.real_vault and (store_clock is not None or os.path.realpath(a.vault) != os.path.realpath(S.REAL_VAULT)):
        say(REAL_VAULT_MISUSE)              # the flag means exactly the real vault, on the system clock, nothing else
        return 2
    try:
        # Without --real-vault the real vault is refused here, by the store itself. With it the store requires an
        # Authorization for every write, and `memory_execute.commit` refuses without the Tier 2 gate built below.
        store = S.MemoryStore(a.vault, allow_real_vault=a.real_vault, clock=store_clock)
    except S.StoreError as e:
        say(REAL_VAULT_REFUSED if e.reason == "real_vault_refused" else f"refused: {e.reason}")
        return 2
    root = scan.output_dir_ok(a.root, forbidden=[a.vault, os.path.dirname(os.path.realpath(a.vault)), a.transcripts])
    if a.status:
        s = RC.startup_check(root, store)
        say(f"proceed={s.proceed} row={s.row} code={s.code}" + (" (first use)" if s.first_use else ""))
        return 0 if s.proceed else 1
    if a.audit:
        r = RC.audit(root, store)
        say(f"verdict={r.verdict} row={r.row} code={r.code}")
        for line in r.facts + r.inferences:
            say("  " + line)
        if r.proposed_record is not None:
            say("  proposed record (not appended): " + r.proposed_record["action"])
        return 0 if r.verdict == "CLEAN" else 1
    if a.reconcile:
        out = EX.reconcile(store=store, receipts_dir=root, ask=ask, say=say, clock=clock)
    else:
        try:
            run_dir = a.run or RV.latest_run(root)
        except R.ReviewError as e:
            say(f"refused: {e}")
            return 1
        build = make_plan_builder(root, run_dir, a.commit, store, a.vault, a.transcripts, clock, monotonic)
        out = EX.commit(make_plan=build, store=store, receipts_dir=root, ask=ask, say=say, clock=clock,
                        monotonic=monotonic, nonce_fn=nonce_fn,
                        tier2=build_tier2(store, root, clock, monotonic, confirmer_factory))
    say(f"{out.status}: {out.code}" + (f" ({', '.join(map(str, out.details))})" if out.details else ""))
    return 0 if out.status in ("committed", "reconciled") else 1


if __name__ == "__main__":
    sys.exit(main())
