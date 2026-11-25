"""Post-hoc documented-default sensitivity, never a replacement for strict EM."""

import json
import math
import re

from .core import canonical
from .evaluation import parse_prediction, score


def documented_defaults(schema: dict) -> dict:
    parameters = schema.get("parameters", {})
    required = set(parameters.get("required") or [])
    found = {}
    for name, prop in parameters.get("properties", {}).items():
        if name in required:
            continue
        kind = prop.get("type")
        origin = "schema.default"
        if "default" in prop:
            value = prop["default"]
        else:
            description = prop.get("description", "")
            # Deliberately narrow grammar: no inferred, conditional or unquoted string defaults.
            if len(re.findall(r"\bdefault\b", description, re.I)) != 1:
                continue
            match = re.search(r"\bThe default value is (?:'([^']*)'|\"([^\"]*)\"|(true|false|-?\d+))(?:[.,]|$)", description)
            if not match:
                continue
            literal = next(g for g in match.groups() if g is not None)
            origin = "description.literal"
            if kind == "string":
                if match.group(3) is not None:
                    continue
                value = literal
            else:
                try:
                    value = json.loads(literal)
                except ValueError:
                    continue
        valid = ((kind == "string" and type(value) is str)
                 or (kind == "boolean" and type(value) is bool)
                 or (kind in ("integer", "int") and type(value) is int)
                 or (kind == "number" and type(value) in (float, int) and math.isfinite(value)))
        if valid:
            found[name] = {"value": value, "source": origin}
    return found


def default_sensitivity(text: str, gold: dict, schema: dict) -> dict:
    strict = score(text, gold)
    result = {"strict_exact_match": strict["exact_match"], "default_aware_exact_match": 0.0,
              "applied_to_prediction": [], "applied_to_gold": [],
              "note": "Post-hoc schema sensitivity only; real tool default behavior is not executed."}
    try:
        pred = parse_prediction(text)
    except (ValueError, TypeError):
        return result
    if pred.get("name") != gold["name"] or schema.get("name") != gold["name"]:
        return result
    defaults = documented_defaults(schema)
    p, g = dict(pred["arguments"]), dict(gold["arguments"])
    for name, entry in defaults.items():
        if name not in p:
            p[name] = entry["value"]
            result["applied_to_prediction"].append(name)
        if name not in g:
            g[name] = entry["value"]
            result["applied_to_gold"].append(name)
    result["documented_defaults"] = defaults
    result["default_aware_exact_match"] = float(canonical(p) == canonical(g))
    return result
