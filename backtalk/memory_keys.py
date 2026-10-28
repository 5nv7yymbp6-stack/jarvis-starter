"""The Memory 2.0 key registry (Phase 7): which logical keys exist, which are single-valued, and what may be renamed to what.

PURE. No file, no model, no clock, no state. Imports only the pure core (backtalk.memory). Nothing live imports it.

A logical key is `subject.attribute` (here subject is always `user`). Two kinds are allowed:
    singleton   one current value at a time (residence, occupation ...): a second claim for the same key is a
                COLLISION to be resolved by the person, or a supersession by the gate, never a second parallel fact
    set         one key per distinct fact, named by a registered prefix (pref_*, goal_*, rule_*, dislike_*)
A key that is neither is refused (`not_in_registry`): a new kind of fact needs a reviewed registry change and a bumped
REGISTRY_VERSION, so keys cannot drift one phrasing at a time. Aliases map a variant name to its canonical key.
A key, once locked in a review, is frozen on the item and is never recomputed from its value.
"""
from backtalk import memory as M

REGISTRY_VERSION = 1
SINGLETONS = ("residence", "occupation", "business_start", "operating_focus", "language", "address_as", "vehicle",
              "new_ventures_priority")
SET_PREFIXES = ("pref_", "goal_", "rule_", "dislike_")
ALIASES = {"home": "residence", "address": "residence", "job": "occupation", "work": "occupation",
           "business_started": "business_start", "started_business": "business_start", "focus_market": "operating_focus",
           "market_focus": "operating_focus", "preferred_language": "language", "shopify_priority": "new_ventures_priority",
           "venture_priority": "new_ventures_priority"}
SINGLETON, SET, UNKNOWN = "singleton", "set", "unknown"


def canonical(attribute) -> str:
    """The canonical attribute for a name or alias (a name that is not an alias is returned unchanged)."""
    return ALIASES.get(attribute, attribute) if isinstance(attribute, str) else attribute


def kind_of(attribute) -> str:
    a = canonical(attribute)
    if a in SINGLETONS:
        return SINGLETON
    if isinstance(a, str) and any(a.startswith(p) and len(a) > len(p) for p in SET_PREFIXES):
        return SET
    return UNKNOWN


def describe(attribute) -> dict:
    a = canonical(attribute)
    return {"kind": kind_of(a), "canonical": a, "version": REGISTRY_VERSION}


def validate_key(attribute, taken=None) -> tuple:
    """() for an acceptable key, else error codes. `taken` maps key -> owner id for keys already claimed."""
    errors = []
    if not isinstance(attribute, str) or not M._NAME_RE.match(attribute):
        return ("invalid_name",)
    if kind_of(attribute) == UNKNOWN:
        errors.append("not_in_registry")
    if canonical(attribute) != attribute:
        errors.append("use_canonical:" + canonical(attribute))
    if taken and canonical(attribute) in taken:
        errors.append("key_taken")
    return tuple(errors)
