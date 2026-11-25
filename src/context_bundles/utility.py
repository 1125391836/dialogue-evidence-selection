"""Train-only structural set-utility regression, serialized as transparent JSON."""

from itertools import combinations
import math

from .core import Budget, Case, canonical
from .selection import bm25, words


FEATURES = ["fraction_blocks", "budget_fraction", "query_coverage", "schema_coverage",
            "relevance_sum", "relevance_max", "relevance_mean", "recency_mean",
            "pair_overlap_mean", "pair_overlap_max", "pair_query_complement",
            "source_diversity", "query_coverage_x_size", "schema_coverage_x_query"]


def training_examples_match(saved: list[dict], computed: list[dict]) -> bool:
    """Permit only numerical roundoff in features, never changed targets or provenance."""
    if len(saved) != len(computed):
        return False
    for a, b in zip(saved, computed):
        if {k: v for k, v in a.items() if k != "features"} != {k: v for k, v in b.items() if k != "features"}:
            return False
        x, y = a.get("features", []), b.get("features", [])
        if len(x) != len(FEATURES) or len(y) != len(FEATURES):
            return False
        if not all(math.isfinite(u) and math.isfinite(v) and math.isclose(u, v, rel_tol=1e-12, abs_tol=1e-12)
                   for u, v in zip(x, y)):
            return False
    return True


class SetFeatures:
    def __init__(self, case: Case, budget: Budget):
        self.case = case
        self.budget = budget
        self.by_id = {e.id: e for e in case.evidence}
        self.terms = {e.id: set(words(e.text)) for e in case.evidence}
        self.query = set(words(case.query))
        properties = case.tool_schema.get("parameters", {}).get("properties", {})
        self.schema = set(words(" ".join(properties)))
        scores = bm25(case.query, case.evidence)
        scale = max(scores.values(), default=1) or 1
        self.scores = {k: v / scale for k, v in scores.items()}
        self.cache = {}

    def __call__(self, ids: frozenset[str]) -> list[float]:
        key = tuple(sorted(ids))
        if key in self.cache:
            return self.cache[key]
        if not ids <= self.by_id.keys():
            raise ValueError("Unknown evidence in utility features")
        chosen = [self.by_id[i] for i in key]
        union = set().union(*(self.terms[i] for i in key)) if key else set()
        qcov = len(union & self.query) / max(1, len(self.query))
        scov = len(union & self.schema) / max(1, len(self.schema))
        rel = [self.scores[i] for i in key]
        overlaps, complements = [], []
        for a, b in combinations(key, 2):
            sa, sb = self.terms[a], self.terms[b]
            overlaps.append(len(sa & sb) / max(1, len(sa | sb)))
            qa, qb = sa & self.query, sb & self.query
            complements.append(min(len(qa - qb), len(qb - qa)) / max(1, len(self.query)))
        max_order = max((e.order for e in self.case.evidence), default=1) or 1
        fraction = len(key) / max(1, len(self.case.evidence))
        result = [fraction, self.budget.cost(chosen) / self.budget.limit, qcov, scov,
                  sum(rel), max(rel, default=0), sum(rel) / max(1, len(rel)),
                  sum(e.order / max_order for e in chosen) / max(1, len(chosen)),
                  sum(overlaps) / max(1, len(overlaps)), max(overlaps, default=0),
                  max(complements, default=0),
                  len({e.session_id for e in chosen}) / max(1, len(chosen)),
                  qcov * fraction, scov * qcov]
        self.cache[key] = result
        return result


def train_predictor(examples: list[dict], alpha: float = 10.0) -> dict:
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if not examples or any(e["split"] != "train" for e in examples):
        raise ValueError("Predictor training accepts only nonempty training-split examples")
    x = np.asarray([e["features"] for e in examples], dtype=float)
    y = np.asarray([e["target"] for e in examples], dtype=float)
    groups = np.asarray([e["group_id"] for e in examples])
    if x.shape[1] != len(FEATURES) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Invalid predictor features or targets")
    unique = sorted(set(groups))
    scores = []
    if len(unique) >= 2:
        for train, valid in GroupKFold(n_splits=min(5, len(unique))).split(x, y, groups):
            estimator = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            estimator.fit(x[train], y[train])
            predicted = np.clip(estimator.predict(x[valid]), 0, 1)
            scores.append({"mse": float(np.mean((predicted - y[valid]) ** 2)),
                           "constant_mse": float(np.mean((np.mean(y[train]) - y[valid]) ** 2)),
                           "validation_groups": sorted(set(groups[valid]))})
    final = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(x, y)
    scaler, ridge = final.steps[0][1], final.steps[1][1]
    return {"schema_version": 1, "kind": "structural_set_utility_ridge_prototype",
            "target": "exact_match", "features": FEATURES,
            "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
            "coefficient": ridge.coef_.tolist(), "intercept": float(ridge.intercept_),
            "alpha": alpha, "training_rows": len(examples), "training_groups": unique,
            "feature_min": x.min(axis=0).tolist(), "feature_max": x.max(axis=0).tolist(),
            "training_case_ids": sorted({e["case_id"] for e in examples}),
            "positive_fraction": float(y.mean()), "group_cv": scores,
            "gate": "exploratory_insufficient_sample" if len(unique) < 30 else "requires_external_validation",
            "constant_target": bool(len(set(y)) == 1),
            "note": "No train-set fit score is evidence of generalization. Features use public observations only."}


def predict(model: dict, features: list[float]) -> float:
    if model["features"] != FEATURES or len(features) != len(FEATURES):
        raise ValueError("Feature schema mismatch")
    value = model["intercept"] + sum(c * (x - m) / s for c, x, m, s in zip(
        model["coefficient"], features, model["mean"], model["scale"]))
    if not math.isfinite(value):
        raise ValueError("Nonfinite utility")
    return min(1.0, max(0.0, value))
