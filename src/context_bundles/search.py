"""Budgeted set search with an explicit utility interface.

An exact utility callback is allowed ONLY in synthetic/oracle diagnostics.
Real test-time use requires a frozen predictor trained without test labels.
These algorithms are standard baselines/search primitives, not novelty claims.
"""

from itertools import combinations
from typing import Callable, Iterable

Set = frozenset[str]
Utility = Callable[[Set], float]


def marginal_greedy(ids: Iterable[str], feasible: Callable[[Set], bool], utility: Utility) -> Set:
    ids = tuple(sorted(set(ids)))
    selected: Set = frozenset()
    while True:
        current = utility(selected)
        options = [selected | {i} for i in sorted(ids) if i not in selected and feasible(selected | {i})]
        if not options:
            return selected
        best = max(options, key=lambda s: (utility(s), tuple(sorted(s))))
        if utility(best) <= current:
            return selected
        selected = best


def bundle_beam(
    ids: Iterable[str], bundles: Iterable[Iterable[str]],
    feasible: Callable[[Set], bool], utility: Utility,
    width: int = 4, max_evaluations: int = 100,
) -> tuple[Set, int]:
    if width < 1 or max_evaluations < 1:
        raise ValueError("Search limits must be positive")
    universe = frozenset(ids)
    moves = {frozenset([i]) for i in universe}
    for bundle in bundles:
        move = frozenset(bundle)
        if not move <= universe:
            raise ValueError("Bundle contains an unknown evidence ID")
        if move:
            moves.add(move)
    ordered_moves = sorted(moves, key=lambda s: (len(s), tuple(sorted(s))))
    cache: dict[Set, float] = {frozenset(): utility(frozenset())}
    beam = [frozenset()]
    while len(cache) < max_evaluations:
        proposals: set[Set] = set(beam)
        for state in beam:
            for move in ordered_moves:
                nxt = state | move
                if feasible(nxt):
                    proposals.add(nxt)
        new = sorted(proposals - cache.keys(), key=lambda s: (len(s), tuple(sorted(s))))
        if not new:
            break
        for state in new:
            if len(cache) == max_evaluations:
                break
            cache[state] = utility(state)
        beam = sorted(
            (s for s in proposals if s in cache),
            key=lambda s: (-cache[s], len(s), tuple(sorted(s))),
        )[:width]
    best = min(cache, key=lambda s: (-cache[s], len(s), tuple(sorted(s))))
    return best, len(cache)


def exhaustive(ids: Iterable[str], feasible: Callable[[Set], bool], utility: Utility) -> Set:
    values = sorted(ids)
    if len(values) > 20:
        raise ValueError("Exhaustive oracle is restricted to at most 20 items")
    possible = (frozenset(c) for n in range(len(values) + 1) for c in combinations(values, n))
    return min((s for s in possible if feasible(s)), key=lambda s: (-utility(s), len(s), tuple(sorted(s))))


def synthetic_demo() -> dict:
    # Deliberately constructed AND utility; proves a code path, not real-world prevalence.
    ids = ["owner_mapping", "contact_mapping", "distractor"]
    def utility(s: Set) -> float:
        return 1.0 if {"owner_mapping", "contact_mapping"} <= s else (0.1 if "distractor" in s else 0.0)
    feasible = lambda s: len(s) <= 2
    greedy = marginal_greedy(ids, feasible, utility)
    beam, calls = bundle_beam(ids, [(ids[0], ids[1])], feasible, utility)
    oracle = exhaustive(ids, feasible, utility)
    return {
        "kind": "synthetic_known_utility_demo_not_model_results",
        "budget_items": 2,
        "greedy": {"selected": sorted(greedy), "utility": utility(greedy)},
        "bundle_beam": {"selected": sorted(beam), "utility": utility(beam), "utility_evaluations": calls},
        "exhaustive": {"selected": sorted(oracle), "utility": utility(oracle)},
    }
