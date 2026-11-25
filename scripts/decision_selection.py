"""Fixed-budget selection of an offer together with the user's decision."""
import re

from context_bundles.selection import pack, select

VALUE = re.compile(r"\b(?:\d+(?:[:.]\d+)?(?:\s*[ap]\.?m\.?)?|one|two|three|four|five|six|seven|eight|nine|ten|today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|noon|midnight)\b", re.I)
DECIDE = re.compile(r"\b(?:yes|yeah|yep|correct|sounds good|that works|that's right|perfect|first|second|earlier|later|earliest|latest|go ahead|please do|that one|i.ll take|book it|no thanks|no thank you)\b", re.I)
REVISE = re.compile(r"\b(?:instead|actually|rather|change|make that|switch|not that|no longer|no[, ])\b",re.I)


def choose(case, budget, method, pivot=None):
    if method == "recent":
        return select(case,"recent",budget)
    history = list(case.evidence)
    if method == "pivot":
        scores = pivot.scores(case)
        return pack([e for _,e in sorted(zip(scores,history),key=lambda t:(t[0],t[1].order),reverse=True)],budget)
    if method not in {"adjacent","decision"}:
        raise ValueError(method)
    groups = []
    bodies = [e.text.partition(": ")[2] for e in history] + [case.query]
    roles = [e.text.partition(":")[0] for e in history] + ["U"]
    for i in range(len(history)):
        if roles[i] != "A" or roles[i+1] != "U" or not VALUE.search(bodies[i]):
            continue
        reply = bodies[i+1]
        priority = 1
        if method == "decision":
            if REVISE.search(reply):
                priority = 2
            elif not DECIDE.search(reply):
                continue
        groups.append((priority,i,history[i:i+2]))
    chosen = []
    ids = set()
    for _,_,group in sorted(groups,reverse=True):
        extended = sorted(chosen+[e for e in group if e.id not in ids],key=lambda e:e.order)
        if budget.fits(extended):
            chosen = extended
            ids.update(e.id for e in group)
            if method == "decision":
                break
    return pack(chosen+[e for e in reversed(history) if e.id not in ids],budget)
