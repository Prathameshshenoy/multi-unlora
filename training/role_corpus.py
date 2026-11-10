"""Pinned misconception-role development corpus, not an official benchmark.

No model-selected examples. Existing evaluation questions and demo concepts are
excluded before deterministic allocation. The source labels are reference
annotations, not independently adjudicated claims.
"""
import csv
import hashlib
import io
import json
import re
import unicodedata
from urllib.request import urlopen

REVISION='f6be04e52bbcb41d4d20daee6358231d4a5015d2'
SOURCE_URL=f'https://raw.githubusercontent.com/sylinrl/TruthfulQA/{REVISION}/TruthfulQA.csv'
SOURCE_SHA256='b8d8ef1e12f98b4f2a9f47abc9765da0640b182b6c5d9b92f0c1a1f2f1e02e5c'
PROTOCOL='disjoint-truthfulqa-role-development-v1'


def canonical_question(text):
    return ' '.join(re.findall(r'\w+',unicodedata.normalize('NFKC',text).casefold()))


def question_id(text):
    return hashlib.sha256(canonical_question(text).encode()).hexdigest()


def source_records(data=None):
    if data is None:
        with urlopen(SOURCE_URL,timeout=45) as response:
            data=response.read()
    if hashlib.sha256(data).hexdigest()!=SOURCE_SHA256:
        raise ValueError('Pinned role corpus checksum mismatch')
    return list(csv.DictReader(io.StringIO(data.decode('utf-8'))))


def validate_split(split):
    seen=set()
    for name in ('train','development','test'):
        keys=[question_id(row['prompt']) for row in split[name]]
        if len(set(keys))!=len(keys) or seen.intersection(keys):
            raise ValueError('Role corpus split overlap')
        seen.update(keys)
    if seen.intersection(split.get('blocked_question_ids',[])):
        raise ValueError('Role corpus overlaps consumed evaluation')


def prepare_corpus(records,*,consumed=(),blocked_question_ids=(),train_n=300,development_n=32,test_n=64):
    blocked={question_id(q) for q in consumed} | set(blocked_question_ids)
    unique={}
    for row in records:
        prompt,good,bad=(row[k].strip() for k in ('Question','Best Answer','Best Incorrect Answer'))
        key=question_id(prompt)
        # All gum/Einstein variants are qualitative probes only. Punctuation,
        # case and whitespace variants of consumed questions are also blocked.
        if key in blocked or re.search(r'\b(gum|chewing|einstein)\b',prompt,re.I):
            continue
        if not prompt or not good or not bad:
            continue
        good_refs={canonical_question(v) for v in row['Correct Answers'].split(';')}
        bad_refs={canonical_question(v) for v in row['Incorrect Answers'].split(';')}
        if canonical_question(bad) in good_refs or canonical_question(good) in bad_refs or canonical_question(good)==canonical_question(bad):
            raise ValueError('Source contains contradictory correct/incorrect reference labels')
        candidate={'prompt':prompt,'good':good,'bad':bad,'question_id':key,'category':row['Category']}
        if key in unique and unique[key]!=candidate:
            raise ValueError('Conflicting duplicate source question')
        unique[key]=candidate
    ordered=sorted(unique.values(),key=lambda r:hashlib.sha256(('role-split-3407:'+r['question_id']).encode()).hexdigest())
    if len(ordered)<train_n+development_n+test_n:
        raise ValueError('Insufficient disjoint role-development records')
    split={'protocol':PROTOCOL,'train':ordered[:train_n],
           'development':ordered[train_n:train_n+development_n],
           'test':ordered[train_n+development_n:train_n+development_n+test_n],
           'blocked_question_ids':sorted(blocked),'eligible_rows':len(ordered),
           'source_sha256':SOURCE_SHA256,'source_url':SOURCE_URL,
           'scope':'Derived internal role-development split. Not an official TruthfulQA score; full-dataset evaluation would overlap training.'}
    validate_split(split)
    return split


def split_fingerprint(split):
    return hashlib.sha256(json.dumps(split,sort_keys=True).encode()).hexdigest()


def load_frozen_corpus():
    from training.role_allocation import ALLOCATION,SPLIT_SHA256
    split=prepare_corpus(source_records(),blocked_question_ids=ALLOCATION['blocked_question_ids'])
    if split_fingerprint(split)!=SPLIT_SHA256:
        raise ValueError('Frozen role allocation changed')
    return split
