"""Stage 0 evaluation. Reads the labeled store, embeds each sample with each
model, and reports separation between YUSUF and the other classes.

    python -m backtalk.evals.speaker.evaluate --models resemblyzer,ecapa
    python -m backtalk.evals.speaker.evaluate --purge-audio

It NEVER chooses a production threshold. The threshold table is a grid of
candidates so the trade-off is visible."""
import argparse
import json
import re
import statistics
import time
from pathlib import Path

import numpy as np

from backtalk.evals.speaker import enroll, metrics
from backtalk.evals.speaker.store import Store

GRID = [round(x, 2) for x in np.arange(0.30, 0.951, 0.05)]
NEG_CLASSES = ("OTHER_SPEAKER", "TV_RECORDED", "SYNTHETIC_TTS")


# ---------------- pure analysis (unit-tested) ----------------

def _pairs(samples, embeds, label):
    return [s for s in samples if s.label == label and s.sample_id in embeds]


def analyze(samples, embeds, existing_profile=None, k_folds=5,
            thresholds=GRID, multi_k=3) -> dict:
    """samples: list[Sample]; embeds: {sample_id: vector} for ONE model.
    existing_profile: an already-enrolled profile vector for this model, or
    None. Returns nested dict of blocks (see metrics.report_block)."""
    yus = _pairs(samples, embeds, "YUSUF")
    out = {"n": {lab: len(_pairs(samples, embeds, lab))
                 for lab in ("YUSUF",) + NEG_CLASSES}}

    def run(score_pos, score_neg_for):
        """score_pos: {sample_id: score}; score_neg_for: label -> {id: score}"""
        res = {}
        pos_items = [(s.duration_s, score_pos[s.sample_id], s.path)
                     for s in yus if s.sample_id in score_pos]
        for neg_label in NEG_CLASSES:
            negs = [(s.duration_s, score_neg_for[neg_label][s.sample_id],
                     s.path) for s in _pairs(samples, embeds, neg_label)
                    if s.sample_id in score_neg_for.get(neg_label, {})]
            if not negs or not pos_items:
                continue
            block = {"all": metrics.report_block(
                [(d, s) for d, s, _ in pos_items],
                [(d, s) for d, s, _ in negs], thresholds)}
            for b in metrics.BUCKETS:
                p = [(d, s) for d, s, _ in pos_items
                     if metrics.bucket(d) == b]
                n = [(d, s) for d, s, _ in negs if metrics.bucket(d) == b]
                if p and n:
                    block[b] = metrics.report_block(p, n, thresholds)
            for path in ("aec", "plain"):
                p = [(d, s) for d, s, pa in pos_items if pa == path]
                n = [(d, s) for d, s, pa in negs if pa == path]
                if p and n:
                    block[f"path:{path}"] = metrics.report_block(
                        p, n, thresholds)
            res[neg_label] = block
        # Yusuf-only distribution per bucket and per path, for context
        res["yusuf_by_bucket"] = {
            b: metrics.describe([s for d, s, _ in pos_items
                                 if metrics.bucket(d) == b])
            for b in metrics.BUCKETS}
        res["yusuf_by_path"] = {
            pa: metrics.describe([s for d, s, p2 in pos_items if p2 == pa])
            for pa in ("aec", "plain")}
        return res

    if existing_profile is not None:
        allsamp = [s for s in samples if s.sample_id in embeds]
        sc = {s.sample_id: enroll.cosine(embeds[s.sample_id],
                                         existing_profile) for s in allsamp}
        out["existing_profile"] = run(
            {s.sample_id: sc[s.sample_id] for s in yus},
            {lab: {s.sample_id: sc[s.sample_id]
                   for s in _pairs(samples, embeds, lab)}
             for lab in NEG_CLASSES})

    # cross-validated enrollment: enroll on the other blocks, score the held-out
    if len(yus) >= 2 * 2:
        folds = enroll.blocks(len(yus), k_folds)
        if len(folds) >= 2:
            pos_c, pos_m = {}, {}
            neg_c = {lab: {} for lab in NEG_CLASSES}
            neg_m = {lab: {} for lab in NEG_CLASSES}
            neg_acc_c = {lab: {} for lab in NEG_CLASSES}
            neg_acc_m = {lab: {} for lab in NEG_CLASSES}
            for held in folds:
                train = [yus[i] for i in range(len(yus)) if i not in held]
                tvec = [embeds[s.sample_id] for s in train]
                prof = enroll.centroid(tvec)
                for i in held:
                    e = embeds[yus[i].sample_id]
                    pos_c[yus[i].sample_id] = enroll.score_centroid(e, prof)
                    pos_m[yus[i].sample_id] = enroll.score_multi(e, tvec,
                                                                 multi_k)
                for lab in NEG_CLASSES:
                    for s in _pairs(samples, embeds, lab):
                        e = embeds[s.sample_id]
                        neg_acc_c[lab].setdefault(s.sample_id, []).append(
                            enroll.score_centroid(e, prof))
                        neg_acc_m[lab].setdefault(s.sample_id, []).append(
                            enroll.score_multi(e, tvec, multi_k))
            for lab in NEG_CLASSES:
                neg_c[lab] = {k: float(np.mean(v))
                              for k, v in neg_acc_c[lab].items()}
                neg_m[lab] = {k: float(np.mean(v))
                              for k, v in neg_acc_m[lab].items()}
            out["cv_centroid"] = run(pos_c, neg_c)
            out["cv_multi"] = run(pos_m, neg_m)
            out["cv_folds"] = len(folds)
    return out


def join_production_scores(samples, log_lines, window_s=8.0, dur_tol=0.2):
    """Match labeled samples to the production shadow scores the running
    voice line already logged for the same utterance (its AEC-path score
    against the DEPLOYED profile). Match = timestamp within window_s and the
    logged 'heard Xs' within dur_tol of the sample's duration.
    Returns {sample_id: score}. Reads a log; changes nothing."""
    events = []            # (epoch, score, dur)
    pending = None
    for ln in log_lines:
        m = re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) ", ln)
        if not m:
            continue
        ts = time.mktime(time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))
        ms = re.search(r"shadow: score ([0-9.]+)", ln)
        if ms:
            pending = [ts, float(ms.group(1)), None]
            continue
        md = re.search(r"\[ears\] heard ([0-9.]+)s", ln)
        if md and pending and pending[2] is None:
            pending[2] = float(md.group(1))
            events.append(tuple(pending))
            pending = None
    out = {}
    for s in samples:
        t = time.mktime(time.strptime(s.timestamp, "%Y-%m-%dT%H:%M:%S"))
        best = None
        for ts, sc, du in events:
            if abs(ts - t) <= window_s and abs(du - s.duration_s) <= dur_tol:
                if best is None or abs(ts - t) < abs(best[0] - t):
                    best = (ts, sc)
        if best:
            out[s.sample_id] = best[1]
    return out


# ---------------- rendering ----------------

def _f(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def render_block(title, block) -> str:
    lines = [f"**{title}**  "
             f"YUSUF n={block['pos'].get('n', 0)} "
             f"(median {_f(block['pos'].get('p50'))}), "
             f"other n={block['neg'].get('n', 0)} "
             f"(median {_f(block['neg'].get('p50'))}); "
             f"EER={_f(block['eer'])} at thr {_f(block['eer_threshold'])}; "
             f"d'={_f(block['d_prime'], 2)}; overlap={_f(block['overlap'], 2)}"]
    lines.append("")
    lines.append("| threshold | false accept | false reject |")
    lines.append("|---|---|---|")
    for r in block["table"]:
        lines.append(f"| {r['threshold']:.2f} | {_f(r['far'], 2)} | "
                     f"{_f(r['frr'], 2)} |")
    return "\n".join(lines)


def render(model_name, res) -> str:
    out = [f"## Model: {model_name}", f"Samples: {res['n']}", ""]
    for key, label in (("existing_profile", "Existing deployed profile"),
                       ("cv_centroid", "Fresh enrollment, centroid "
                        f"(cross-validated, {res.get('cv_folds', '?')} blocks)"),
                       ("cv_multi", "Fresh enrollment, multi-embedding "
                        "(top-3 mean, cross-validated)")):
        if key not in res:
            continue
        out.append(f"### {label}")
        sect = res[key]
        for neg in NEG_CLASSES:
            if neg not in sect:
                continue
            out.append(f"#### YUSUF vs {neg}")
            for slice_name, blk in sect[neg].items():
                out.append(render_block(slice_name, blk))
                out.append("")
        out.append("YUSUF score by duration bucket: " + json.dumps(
            {b: {k: round(v, 3) if isinstance(v, float) else v
                 for k, v in d.items()} for b, d in
             sect["yusuf_by_bucket"].items()}))
        out.append("")
    return "\n".join(out)


# ---------------- I/O ----------------

def compute_embeddings(store, model, cache_dir):
    """{sample_id: vector}, {sample_id: seconds}. Cached per model name so a
    purge of the audio keeps the results reproducible."""
    cache = Path(cache_dir) / f"emb_{model.name}.npz"
    have = {}
    times = {}
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        have = dict(z["emb"].item())
        times = dict(z["times"].item())
    todo = [s for s in store.load() if s.sample_id not in have]
    warmed = False
    for s in todo:
        pcm = store.audio(s.sample_id)
        if pcm is None:
            continue
        if not warmed:
            try:
                model.embed(pcm)          # warm-up, not timed
            except Exception:
                pass
            warmed = True
        t0 = time.perf_counter()
        try:
            have[s.sample_id] = model.embed(pcm)
        except Exception as e:          # counted, never silently dropped
            print(f"[embed failed] {model.name} {s.sample_id}: "
                  f"{type(e).__name__}")
            continue
        times[s.sample_id] = time.perf_counter() - t0
    if todo:
        np.savez(cache, emb=np.array(have, dtype=object),
                 times=np.array(times, dtype=object))
    return have, times


def latency(times: dict) -> dict:
    v = sorted(times.values())
    if not v:
        return {}
    return {"n": len(v), "median_s": statistics.median(v),
            "p90_s": v[min(len(v) - 1, int(len(v) * 0.9))],
            "max_s": v[-1]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="resemblyzer")
    ap.add_argument("--dir", default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--purge-audio", action="store_true",
                    help="delete kept WAVs and exit")
    ap.add_argument("--join-log", default=None,
                    help="also report the production shadow scores for the "
                         "same samples, read from this log file")
    a = ap.parse_args(argv)
    store = Store(Path(a.dir) if a.dir else None)
    if a.purge_audio:
        print(f"purged {store.purge_audio()} audio file(s); metadata and "
              f"embeddings kept")
        return 0
    from backtalk.evals.speaker import models as M
    samples = store.load()
    print(f"{len(samples)} samples in {store.root}")
    report = []
    for name in [n.strip() for n in a.models.split(",") if n.strip()]:
        try:
            model = M.load(name, store.root / "models" / name)
        except M.ModelUnavailable as e:
            print(f"[skip] {name}: {e}")
            continue
        emb, times = compute_embeddings(store, model, store.root)
        prof = None
        if name == "resemblyzer":
            from backtalk import voiceid
            if voiceid.PROFILE_PATH.exists():
                prof = np.load(voiceid.PROFILE_PATH)   # read-only
        res = analyze(samples, emb, prof, a.folds)
        res["latency"] = latency(times)
        report.append(render(name, res))
        report.append(f"Latency ({name}, this machine, CPU): "
                      f"{json.dumps(res['latency'])}\n")
    if a.join_log:
        lines = Path(a.join_log).read_text(errors="replace").splitlines()
        joined = join_production_scores(samples, lines)
        pos = [joined[s.sample_id] for s in samples
               if s.label == "YUSUF" and s.sample_id in joined]
        neg = [joined[s.sample_id] for s in samples
               if s.label == "OTHER_SPEAKER" and s.sample_id in joined]
        report.append("## Production shadow scores for the same utterances "
                      "(deployed profile, AEC path)")
        report.append(f"matched YUSUF={len(pos)} OTHER={len(neg)}")
        if pos and neg:
            blk = metrics.report_block([(0, x) for x in pos],
                                       [(0, x) for x in neg], GRID)
            report.append(render_block("all matched", blk))
    text = "\n".join(report)
    rd = store.root / "reports"
    rd.mkdir(exist_ok=True)
    out = rd / (time.strftime("%Y%m%d-%H%M%S") + ".md")
    out.write_text(text)
    print(text)
    print(f"\nreport saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
