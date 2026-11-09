"""Bounded, unscored free-form diagnostics; no automated safety claims."""
import json
import threading
from pathlib import Path

from evaluation.behavior_cases import CASES, BEHAVIOR_PROTOCOL, model_review_eligible

CASE_IDS = ('benign_sky', 'benign_multiplication', 'false_premise_nobel', 'sensitive_phishing_defense')


def run_behavior_diagnostic(manifest_path='/vol/runs/paired_hall_v1/manifest.json'):
    from demo.backends import GPUBackend
    from inference.runtime import validate_artifacts, sha256_file
    from evaluation.quality import fingerprint
    output = Path(manifest_path).parent/'behavior_report.json'
    if output.exists():
        raise RuntimeError('Behavior diagnostic already exists; preserve the original outcome')
    manifest = validate_artifacts(manifest_path)
    backend = GPUBackend(manifest_path=manifest_path)
    rows = []
    report = {'protocol': BEHAVIOR_PROTOCOL, 'manifest_fingerprint': fingerprint(manifest),
              'manifest_sha256': sha256_file(manifest_path), 'max_new_tokens':96,
              'case_ids':list(CASE_IDS), 'independently_reviewed':False,
              'accuracy_improvement_verified':False, 'safety_improvement_verified':False,
              'scope':'Four predeclared benign/false-premise diagnostics. No unsafe cases executed.',
              'cases':rows}
    for case_id in CASE_IDS:
        case = next(case for case in CASES if case.id == case_id)
        if case.category == 'unsafe' or case.prompt is None:
            raise RuntimeError('This diagnostic accepts only public benign fixtures')
        complete = None
        for event, payload in backend.stream(case.prompt,96,threading.Event()):
            if event == 'complete':
                complete = payload
            elif event == 'error':
                raise RuntimeError('Model generation failed; inspect private service logs')
        if complete is None:
            raise RuntimeError('Incomplete behavior generation')
        rows.append({'case_id':case.id,'category':case.category,'prompt':case.prompt,
                     'review_checks':list(case.review_checks),'generation':complete,
                     'eligible_for_independent_review':{
                         branch:model_review_eligible(value,simulated=complete.get('simulated',True))
                         for branch,value in complete.items() if isinstance(value,dict) and 'source' in value}})
        # Preserve partial diagnostic evidence if the remaining job times out.
        report['completed_cases'] = len(rows)
        output.write_text(json.dumps(report,indent=2,allow_nan=False))
    report['generation_complete'] = True
    output.write_text(json.dumps(report,indent=2,allow_nan=False))
    return report
