"""Conservative streamed sentence boundary; returns an unmodified raw prefix.

This is a generation-length control, not factuality filtering. Defer ambiguous
numeric endings and common abbreviations instead of cutting a partial answer.
The token budget remains the hard bound when no clear boundary is recognized.
"""
import re

_END=re.compile(r'[.!?](?:[\"\u201d\u2019\)\]]+)?(?=\s|$)')
_ABBREVIATION=re.compile(r'\b(?:mr|mrs|ms|dr|prof|sr|jr|st|vs|etc|e\.g|i\.e)\.$',re.I)


def first_sentence_end(text):
    for match in _END.finditer(text):
        prefix=text[:match.end()].rstrip('\"\u201d\u2019)]')
        if text[match.start()]=='.':
            if prefix.endswith('..') or _ABBREVIATION.search(prefix) or re.search(r'\b(?:[A-Za-z]\.)+$',prefix):continue
            if re.fullmatch(r'\s*\d+\.',prefix):continue
            if match.end()==len(text) and re.search(r'\d\.$',prefix):continue
            word=prefix.split()[-1] if prefix.split() else ''
            if match.end()==len(text) and ('://' in word or word.startswith('www.')):continue
        return match.end()
    return None
