"""Narrow, explicit safety precedence for the projection router.

This rule recognizes direct explosive-construction, physical-violence and
crime-concealment requests. It is neither a
comprehensive harmful-intent classifier nor evidence of a safe model response.
"""
import re
import unicodedata

ROUTER_VERSION = 'repe-explicit-safety-v2'

# Match actionable requests, rather than every mention of explosives. The
# optional raw user prompt avoids treating chat-template instructions as intent.
_CONSTRUCTION = (
    r'(?:make|making|build(?:ing)?|manufactur(?:e|ing)|assembl(?:e|ing)|construct(?:ing)?)'
    r'\s+(?:me\s+)?(?:an?\s+)?(?:(?:homemade|working|pipe|improvised)\s+){0,2}'
    r'(?:bombs?|explosive\s+devices?|improvised\s+explosive\s+devices?)\b'
    r'(?![\s-]*(?:proof|calorimeters?|shelters?|safe|harmless)\b)'
)
_ACTIONABLE_PREFIX = (
    r'(?:\bhow\s+(?:to|(?:can|do|could|would|should)\s+(?:i|we|you))\s+'
    r'|\b(?:give|show|write|provide|list)\b[^.!?\n]{0,40}?'
    r'\b(?:steps|instructions|guide|recipe)\s+(?:to|for)\s+'
    r'|^\s*(?:please\s+)?)'
)
_RULES = (
    ('explosive_construction', _CONSTRUCTION),
    ('crime_concealment',
     r'(?:hid(?:e|ing)|conceal(?:ing)?|dispos(?:e|ing)\s+of|get(?:ting)?\s+rid\s+of)'
     r'\s+(?:a\s+|the\s+)?(?:(?:dead|human)\s+)?(?:body|corpse)\b'
     r'(?![\s-]*(?:elements?|tags?|panels?|seams?|scars?|parts?|text|html)\b)'),
    ('physical_violence',
     r'(?:kill(?:ing)?|murder(?:ing)?|poison(?:ing)?|assault(?:ing)?)'
     r'\s+(?:(?:a|the)\s+)?(?:someone|somebody|people|person|victim|neighbor|neighbour)\b'),
)
_REQUESTS = tuple((name, re.compile(_ACTIONABLE_PREFIX + pattern, re.IGNORECASE))
                  for name, pattern in _RULES)


def explicit_safety_intent(prompt):
    text = unicodedata.normalize('NFKC', prompt)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Cf')
    return next((name for name, pattern in _REQUESTS if pattern.search(text)), None)


def prioritize_explicit_intent(prompt, alpha_hall, alpha_jail):
    """Preserve the total; allocate at least 75% to jailbreak on a rule match."""
    if explicit_safety_intent(prompt) is None:
        return alpha_hall, alpha_jail
    total = alpha_hall + alpha_jail
    if total == 0 or alpha_jail >= .75 * total:
        return alpha_hall, alpha_jail
    # Subtraction avoids exceeding the original combined correction strength.
    hall = .25 * total
    return hall, total - hall
