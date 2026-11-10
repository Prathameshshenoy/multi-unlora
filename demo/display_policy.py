"""Conservative display restrictions for an anti-expert research interface.

This is a presentation boundary for recognized weapon/explosive requests, not a
comprehensive intent classifier, a replacement for moderation, or evidence that
subtraction made a model refuse. Dangerous raw model text must never be streamed
before this boundary. A guard fallback is always labelled separately from model
output. Production publication still requires independent safety evaluation.
"""
import re
import unicodedata

POLICY_VERSION = 'weapons-display-v2'
BRANCHES = {'baseline','hallucination','jailbreak','unlora'}
RESTRICTED_TERMS = re.compile(
    r'\b(?:bombs?|explosives?|explosive\s+devices?|detonators?|detonate|'
    r'improvised\s+(?:explosive\s+)?devices?|ieds?|firearms?|weapons?|'
    r'hand\s*grenades?|pipe\s*bombs?)\b', re.IGNORECASE)
GUARD_REFUSAL = ("I can’t help with instructions for making weapons or explosives. "
                 "I can help with safety or non-actionable background information.")
# Only complete, independently harmless short refusals may pass through for a
# restricted request. Prefix matching is insufficient: a refusal can be followed
# by dangerous instructions. Everything else remains withheld, including long
# or ambiguous text. These are output checks, never ASR or quality labels.
APPROVED_REFUSALS = frozenset({
    "i can't help with that.", "i cannot help with that.",
    "i can't help with that request.", "i cannot help with that request.",
    "i can't assist with that request.", "i cannot assist with that request.",
    "sorry, i can't help with that.", "i'm sorry, but i can't help with that request.",
    "i cannot provide instructions for making a bomb.",
    "i can't provide instructions for making a bomb.",
    "i cannot help with instructions for making weapons or explosives.",
    "i cannot provide information or guidance on illegal or harmful activities. can i help you with something else?",
    "i cannot provide information or guidance on illegal or harmful activities, including making a bomb. can i help you with something else?",
    "i can't help you with that request. is there anything else i can help you with?",
    "i cannot provide instructions on how to make a bomb. is there anything else i can help you with?",
    "i can't help with instructions for making weapons or explosives.",
})


def _normalize(value):
    value = unicodedata.normalize('NFKC',value).replace('’',"'")
    value = ''.join(char for char in value if unicodedata.category(char) != 'Cf')
    return ' '.join(value.casefold().split())


def assess_prompt(prompt):
    if not isinstance(prompt,str):
        raise ValueError('Prompt must be text')
    restricted = bool(RESTRICTED_TERMS.search(_normalize(prompt)))
    return {'restricted':restricted,
            'category':'weapons_or_explosives' if restricted else None,
            'reason':('Raw weapon or explosive instructions are not displayed. Adapter influence can still be inspected through correction statistics.' if restricted else ''),
            'policy_version':POLICY_VERSION}


def present_response(policy, branch, text):
    if branch not in BRANCHES or not isinstance(text,str):
        raise ValueError('Unknown branch or non-text model output')
    if not isinstance(policy,dict) or type(policy.get('restricted')) is not bool:
        raise ValueError('Explicit display assessment is required')
    if not policy['restricted']:
        return {'text':text,'source':'model','withheld':False,'reason':''}
    reason = policy.get('reason') or 'Raw output is restricted by the display policy.'
    if branch in {'hallucination','jailbreak'}:
        return {'text':'','source':'withheld','withheld':True,'reason':reason}
    if _normalize(text) in APPROVED_REFUSALS:
        return {'text':text,'source':'model','withheld':False,'reason':''}
    if branch=='unlora':
        return {'text':GUARD_REFUSAL,'source':'display_guard','withheld':True,
                'reason':'The raw corrected response was withheld. This refusal is supplied by the display guard, not demonstrated by subtraction.'}
    return {'text':'','source':'withheld','withheld':True,'reason':reason}
