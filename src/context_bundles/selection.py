"""Transparent baselines. Rule pairs are NOT the proposed learned method."""

from collections import Counter as WordCounter
from itertools import combinations
import math
import re

from .core import Budget, Case, Evidence


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text.lower())


def bm25(query: str, evidence: tuple[Evidence, ...]) -> dict[str, float]:
    counts = [WordCounter(words(e.text)) for e in evidence]
    n = len(counts)
    if not n:
        return {}
    avg = sum(sum(c.values()) for c in counts) / n or 1
    df = WordCounter(w for c in counts for w in c)
    scores = {}
    for e, c in zip(evidence, counts):
        length = sum(c.values())
        score = 0.0
        for w in sorted(set(words(query))):
            tf = c[w]
            if tf:
                idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
                score += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * length / avg))
        scores[e.id] = score
    return scores


def lexical_pairs(evidence: tuple[Evidence, ...], max_pairs: int = 100) -> list[tuple[str, str]]:
    """Sparse entity-like overlap prior, not ground-truth dependency detection.

    Uses repeated identifiers, quoted spans and capitalized words. It can miss
    paraphrases and produce false links. The same-session flag is not a label.
    """
    if max_pairs < 0:
        raise ValueError("max_pairs must be nonnegative")
    pattern = r'\b[A-Z][a-z]{2,}\b|\b\w*[0-9]\w*\b|"[^"\n]{2,50}"'
    stop = {"the", "this", "that", "please", "could", "would", "user", "assistant"}
    keys = {e.id: {s.lower() for s in re.findall(pattern, e.text)} - stop for e in evidence}
    candidates = []
    for a, b in combinations(evidence, 2):
        overlap = keys[a.id] & keys[b.id]
        if overlap:
            candidates.append((len(overlap), a.id, b.id))
    candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
    return [(a, b) for _, a, b in candidates[:max_pairs]]


def pack(order: list[Evidence], budget: Budget) -> list[Evidence]:
    selected: list[Evidence] = []
    for e in order:
        if budget.fits(selected + [e]):
            selected.append(e)
    return sorted(selected, key=lambda e: (e.order, e.id))


def select(case: Case, method: str, budget: Budget, max_pairs: int = 100) -> list[Evidence]:
    if method == "full":
        # Reference only: explicitly not a same-budget method.
        return list(case.evidence)
    if method == "recent":
        return pack(sorted(case.evidence, key=lambda e: (-e.order, e.id)), budget)
    if method in {"schema_bm25", "schema_history", "half_recent"}:
        properties = case.tool_schema.get("parameters", {}).get("properties", {})
        schema_words = " ".join(re.sub(r"([a-z])([A-Z])", r"\1 \2", key).replace("_", " ")
                                + " " + spec.get("description", "") for key, spec in properties.items())
        scores = bm25(case.query + " " + schema_words, case.evidence)
        ranked = sorted(case.evidence, key=lambda e: (-scores[e.id], -e.order, e.id))
        if method == "half_recent":
            minimum = budget.cost([])
            recent_budget = Budget(max(minimum, budget.limit // 2), budget.counter)
            recent = select(case, "recent", recent_budget)
            used = {piece.id for piece in recent}
            return pack(recent + [piece for piece in ranked if piece.id not in used], budget)
        if method == "schema_history":
            # Fuse lexical relevance, recency and user-provided evidence by rank.
            # All three ranks are relative to this history and require no fitted weights.
            recent = sorted(case.evidence, key=lambda e: -e.order)
            user_first = sorted(recent, key=lambda e: not e.text.casefold().startswith("user:"))
            fusion = WordCounter()
            for order in (ranked, recent, user_first):
                for rank, piece in enumerate(order, start=1):
                    fusion[piece.id] += 1 / rank
            ranked.sort(key=lambda e: (-fusion[e.id], -e.order, e.id))
        return pack(ranked, budget)
    scores = bm25(case.query, case.evidence)
    ranked = sorted(case.evidence, key=lambda e: (-scores[e.id], -e.order, e.id))
    if method == "bm25":
        return pack(ranked, budget)
    if method != "rule_pairs":
        raise ValueError(f"Unknown method: {method}")
    pairs = lexical_pairs(case.evidence, max_pairs)
    by_id = {e.id: e for e in case.evidence}
    selected: list[Evidence] = []
    present: set[str] = set()
    # Pick one strongest lexical neighbor alongside each ranked seed, when it fits.
    for e in ranked:
        if e.id in present:
            continue
        neighbors = [b if a == e.id else a for a, b in pairs if e.id in (a, b)]
        neighbors.sort(key=lambda i: (-scores[i], i))
        bundle = [e]
        if neighbors and neighbors[0] not in present:
            bundle.append(by_id[neighbors[0]])
        if not budget.fits(selected + bundle):
            bundle = [e]
        if budget.fits(selected + bundle):
            selected.extend(bundle)
            present.update(x.id for x in bundle)
    return sorted(selected, key=lambda e: (e.order, e.id))
