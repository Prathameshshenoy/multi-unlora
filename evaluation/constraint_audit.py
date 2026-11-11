"""Closed-book diagnosis of a consumed shared error; never an accuracy claim."""
import json
from pathlib import Path
import threading
import config
from evaluation.freeform_review import packet_hash
from evaluation.freeform_trial import MANIFEST

PARENT_SHA='b6513a2cb245a40fe89a07c13bbb2fba18c2670de8b6eeea4b16cabd0c5ea86c'
PROBES=(
 ('rank','Name the third-smallest sovereign country by land area.'),
 ('relative_rank','Which sovereign country is immediately larger than Monaco by land area?'),
 ('named_area','What is Nauru\'s land area in square kilometres?'),
 ('pairwise_area','Which country is smaller in area: Nauru or Tuvalu?'),
 ('vatican_threshold','Is Vatican City at least one square mile in area?'),
 ('monaco_threshold','Is Monaco at least one square mile in area?'),
 ('explicit_exclusion','Exclude countries with areas below one square mile. What is the smallest remaining sovereign country?'),
 ('metric_threshold','Which sovereign country is the smallest with an area of at least 2.59 square kilometres?'))


def audit_constraint():
    from demo.backends import GPUBackend
    from inference.prompts import comparison_prompt,BRIEF_PROTOCOL
    from inference.runtime import sha256_file
    parent=json.loads((Path(config.RUNS_DIR)/'selective_subtraction_v3/confirmation/generation_report.json').read_text())
    if packet_hash(parent)!=PARENT_SHA or parent['manifest_sha256']!=sha256_file(MANIFEST):
        raise ValueError('Constraint diagnostic source changed')
    before=[r for r in parent['records'] if r['prompt']=='What is the smallest country in the world that is at least one square mile in area?']
    if len(before)!=1:raise ValueError('Expected one consumed source question')
    backend=GPUBackend(manifest_path=MANIFEST,use_cache=False)
    report={'protocol':'consumed-country-constraint-diagnostic-v1','phase':'diagnostic','status':'running',
        'scope':'Eight closed-book diagnostic probes plus one reused consumed base answer. Some probes name the target entity or change the question. Not comparative answer accuracy, independent confirmation, retrieval or a serving correction.',
        'manifest_fingerprint':parent['manifest_fingerprint'],'allocation_sha256':parent['allocation_sha256'],
        'source_generation_sha256':PARENT_SHA,'probe_spec_sha256':packet_hash(PROBES),'prompt_variant':BRIEF_PROTOCOL,
        'max_new_tokens':48,'sentence_limit':1,'records':[{'probe':'original_reused','prompt':before[0]['prompt'],'output':before[0]['outputs']['baseline'],'reused':True}],
        'production_quality_verified':False}
    for name,prompt in PROBES:
        done=None
        for event,data in backend.stream(comparison_prompt(prompt,'brief'),48,threading.Event(),sentence_limit=1,_branches=('baseline',)):
            if event=='error':raise RuntimeError('Constraint probe failed')
            if event=='complete':done=data
        if done is None or done['simulated'] is not False:raise RuntimeError('Incomplete base probe')
        report['records'].append({'probe':name,'prompt':prompt,'output':done['baseline'],'reused':False})
    report['status']='complete'
    with (Path(config.RUNS_DIR)/'selective_subtraction_v3/constraint_audit.json').open('x') as f:
        json.dump(report,f,indent=2,allow_nan=False)
    return report
