"""Conservative engineering gates, not statistical or publication guarantees."""

import math

from .selection import select
from .utility import FEATURES


def promotion_gate(model: dict, minimum_groups: int = 30) -> dict:
    reasons = []
    if len(model.get("training_groups", [])) < minimum_groups:
        reasons.append("insufficient_independent_training_groups")
    if model.get("constant_target", True):
        reasons.append("constant_training_target")
    cv = model.get("group_cv", [])
    model_mse = sum(r["mse"] for r in cv) / len(cv) if cv else None
    constant_mse = sum(r["constant_mse"] for r in cv) / len(cv) if cv else None
    if not cv or not math.isfinite(model_mse) or not math.isfinite(constant_mse) or model_mse >= constant_mse:
        reasons.append("no_group_cv_gain_over_constant")
    if "feature_min" not in model or "feature_max" not in model:
        reasons.append("missing_training_support")
    return {"allowed_for_guarded_pilot": not reasons, "reasons": reasons,
            "minimum_groups": minimum_groups, "group_cv_mse": model_mse,
            "constant_cv_mse": constant_mse,
            "note": "Engineering minimum only. Passing is not evidence of statistical significance or readiness for publication."}


def support_violations(model: dict, features: list[float]) -> list[str]:
    if model.get("features") != FEATURES or len(features) != len(FEATURES):
        raise ValueError("Feature schema mismatch")
    lower, upper = model.get("feature_min"), model.get("feature_max")
    if lower is None or upper is None:
        return ["missing_training_support"]
    if len(lower) != len(FEATURES) or len(upper) != len(FEATURES):
        raise ValueError("Invalid training support")
    return [name for name, value, lo, hi in zip(FEATURES, features, lower, upper)
            if not all(math.isfinite(x) for x in (value, lo, hi)) or lo > hi
            or value < lo - 1e-9 or value > hi + 1e-9]


def guarded_choice(case, budget, model, proposed_ids, features):
    """Reject untrusted proposals and emit a traceable, same-budget BM25 fallback."""
    gate = promotion_gate(model)
    violations = support_violations(model, features)
    reasons = list(gate["reasons"])
    if violations:
        reasons.append("outside_training_support")
    if not proposed_ids and case.evidence:
        reasons.append("empty_proposal_with_available_history")
    by_id = {e.id: e for e in case.evidence}
    if not set(proposed_ids) <= by_id.keys():
        raise ValueError("Unknown proposed evidence")
    selected = sorted((by_id[i] for i in proposed_ids), key=lambda e: (e.order, e.id))
    if not budget.fits(selected):
        reasons.append("proposal_over_budget")
    if reasons:
        selected = select(case, "bm25", budget)
    return selected, {"gate": gate, "fallback": "bm25" if reasons else None,
                      "reasons": reasons, "support_violations": violations,
                      "support_note": "Coordinate ranges are necessary screening, not a joint-distribution guarantee."}
