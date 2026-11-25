"""Bounded, query-driven recovery from an immutable original case."""

from dataclasses import dataclass, replace

from .core import Budget, Case, Evidence, messages
from .selection import bm25, select


@dataclass(frozen=True)
class HistoryWindow:
    case: Case
    visible: tuple[Evidence, ...]
    budget: Budget

    @classmethod
    def compress(cls, case: Case, budget: Budget, method: str = "recent"):
        if method not in {"recent", "bm25", "rule_pairs"}:
            raise ValueError("Choose a bounded compression method")
        if len({e.id for e in case.evidence}) != len(case.evidence):
            raise ValueError("Evidence IDs must be unique within a case")
        return cls(case, tuple(select(case, method, budget)), budget)

    def prompt(self):
        return messages(self.case, list(self.visible))

    def recover(self, query: str, top_k: int, read_limit: int):
        """Read limit counts the serialized bundle admitted from the archive."""
        if min(top_k, read_limit) < 0:
            raise ValueError("Recovery limits must be nonnegative")
        scores = bm25(query, self.case.evidence)
        visible_ids = {e.id for e in self.visible}
        hidden = sorted((e for e in self.case.evidence if e.id not in visible_ids
                         and scores[e.id] > 0), key=lambda e: (-scores[e.id], -e.order, e.id))
        keep = list(self.visible)
        added = []
        for candidate in hidden:
            if len(added) == top_k:
                break
            if self.budget.cost(added + [candidate]) > read_limit:
                continue
            trial = keep.copy()
            removable = sorted((e for e in keep if scores[e.id] < scores[candidate.id]),
                               key=lambda e: (scores[e.id], e.order, e.id))
            for old in removable:
                if self.budget.fits(trial + added + [candidate]):
                    break
                trial.remove(old)
            if self.budget.fits(trial + added + [candidate]):
                keep = trial
                added.append(candidate)
        visible = tuple(sorted(keep + added, key=lambda e: (e.order, e.id)))
        window = HistoryWindow(replace(self.case, query=query), visible, self.budget)
        trace = {
            "before": [e.id for e in self.visible], "after": [e.id for e in visible],
            "recovered": [e.id for e in added], "evicted": [e.id for e in self.visible if e not in keep],
            "before_cost": self.budget.cost(self.visible), "after_cost": self.budget.cost(visible),
            "read_cost": self.budget.cost(added) if added else 0,
        }
        return window, trace
