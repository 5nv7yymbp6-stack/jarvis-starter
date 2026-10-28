"""Enrollment math for the experiments. Pure numpy. Experimental profiles are
built in memory or under the eval data dir, never over the production
voiceprint."""
import numpy as np


def unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n else v


def cosine(a, b) -> float:
    return float(np.dot(unit(a), unit(b)))


def centroid(embs) -> np.ndarray:
    """One profile: the normalized mean of unit embeddings."""
    return unit(np.mean([unit(e) for e in embs], axis=0))


def score_centroid(e, profile) -> float:
    return cosine(e, profile)


def score_multi(e, embs, k: int = 3) -> float:
    """Multi-embedding enrollment: mean of the top-k cosines against the
    individual enrollment embeddings (keeps distinct 'modes' of a voice
    apart instead of averaging them away)."""
    sims = sorted((cosine(e, x) for x in embs), reverse=True)
    return float(np.mean(sims[:max(1, min(k, len(sims)))]))


def blocks(n: int, k: int) -> list[list[int]]:
    """k contiguous index blocks over range(n). Contiguous, not interleaved:
    neighbouring samples share a session and room, so leaving a whole block
    out is the stricter test. Never returns an empty block."""
    k = max(1, min(k, n))
    edges = [round(i * n / k) for i in range(k + 1)]
    return [list(range(edges[i], edges[i + 1])) for i in range(k)
            if edges[i + 1] > edges[i]]
