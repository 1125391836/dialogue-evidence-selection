"""Cluster-aware paired descriptive inference, with explicit small-sample gates."""
from collections import defaultdict
import random


def paired_cluster_bootstrap(rows: list[dict], left: str, right: str, metric: str = "exact_match",
                             repetitions: int = 2000, seed: int = 42) -> dict:
    if repetitions < 1:
        raise ValueError("Positive bootstrap repetitions required")
    by_case = defaultdict(lambda: defaultdict(list))
    groups = {}
    for row in rows:
        if row["method"] in (left, right):
            by_case[row["case_id"]][row["method"]].append(row[metric])
            groups[row["case_id"]] = row["group_id"]
    differences = defaultdict(list)
    missing = []
    for case_id, methods in by_case.items():
        if left not in methods or right not in methods:
            missing.append(case_id)
            continue
        delta = sum(methods[left]) / len(methods[left]) - sum(methods[right]) / len(methods[right])
        differences[groups[case_id]].append(delta)
    keys = sorted(differences)
    if not keys:
        return {"left": left, "right": right, "paired_cases": 0, "missing_cases": missing}
    values = [v for k in keys for v in differences[k]]
    rng = random.Random(seed)
    boot = []
    for _ in range(repetitions):
        sample = [v for k in rng.choices(keys, k=len(keys)) for v in differences[k]]
        boot.append(sum(sample) / len(sample))
    boot.sort()
    return {"left": left, "right": right, "metric": metric,
            "paired_cases": len(values), "independent_groups": len(keys),
            "mean_difference": sum(values) / len(values),
            "interval_95": [boot[int(0.025 * (repetitions - 1))], boot[int(0.975 * (repetitions - 1))]],
            "missing_cases": missing, "bootstrap_repetitions": repetitions,
            "interpretation": "descriptive_only_too_few_groups" if len(keys) < 20 else "check_preregistered_inference_protocol"}
