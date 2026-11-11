"""Run bounded Modal stages and record conservative budget reservations.

Cloud reservations are deliberately not refunded on failures: unknown cloud activity
must never silently expand the approved budget. CPU preflight reserves $0.25;
GPU time reserves $0.001/second, above listed A100+requested CPU/RAM rates.
This is a workload ceiling, not an invoice or a provider billing API.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import os
import shutil
import uuid

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RESERVATIONS = {'preflight': .25, 'train': 3.6, 'evaluate': 1.2, 'tune-confirm': 1.2, 'serve-local': 1.0, 'paired-train': 1.8, 'paired-evaluate': 1.8, 'behavior': .6, 'anchored-train': 1.8, 'anchored-evaluate': 1.8, 'anchored-long-train': 1.8, 'anchored-long-evaluate': 1.8, 'short-answers': 1.2, 'data-audit': .25, 'matched-train': 1.8, 'matched-evaluate': 1.8, 'negative-train': 1.8, 'negative-evaluate': 1.8, 'role-repair': 2.4, 'content-delta': 1.8, 'freeform-dev': .9, 'reference-dev': 1.2, 'strength-dev': .6, 'divergence-audit': .3, 'support-dev': .9, 'support-confirm': .6, 'selective-dev': .5}
SERVE_RESERVATIONS = {3: .3, 5: .5, 10: .8, 15: 1.0}


def reserve(stage, *, minutes=15):
    import fcntl
    state = ROOT/'.runtime'
    state.mkdir(exist_ok=True)
    with (state/'budget.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path = state/'budget.json'
        ledger = json.loads(path.read_text()) if path.exists() else {
            'approved_usd':10, 'prior_cpu_preflight_reserve_usd':.5, 'reservations':[]}
        spent = ledger['prior_cpu_preflight_reserve_usd'] + sum(r['reserved_usd'] for r in ledger['reservations'])
        cost = SERVE_RESERVATIONS[minutes] if stage=="serve-local" else RESERVATIONS[stage]
        if spent+cost > ledger['approved_usd']:
            raise RuntimeError(f'Conservative budget ceiling reached: ${spent:.2f} reserved; stage needs ${cost:.2f}')
        ledger['reservations'].append({'stage':stage,'reserved_usd':cost,
            'at':datetime.now(timezone.utc).isoformat()})
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(ledger,indent=2))
        temporary.replace(path)
        print(f'Budget reservation: ${spent+cost:.2f} / ${ledger["approved_usd"]:.2f}',flush=True)


def dispatch_stage(engine, stage, steps):
    """Invoke exactly one budgeted cloud function; serving is handled separately."""
    jobs={
        'selective-dev':(engine.develop_selective_subtraction,()),
        'support-confirm':(engine.confirm_reference_support,()),
        'support-dev':(engine.develop_reference_support,()),
        'divergence-audit':(engine.audit_factual_divergence,()),
        'strength-dev':(engine.calibrate_factual_strength,()),
        'reference-dev':(engine.develop_factual_reference,()),
        'freeform-dev':(engine.develop_freeform,()),
        'content-delta':(engine.repair_content_delta,()),
        'role-repair':(engine.repair_roles,()),
        'preflight':(engine.preflight,()),
        'data-audit':(engine.audit_training_data,()),
        'matched-train':(engine.train_matched,()),
        'matched-evaluate':(engine.evaluate_matched,()),
        'negative-train':(engine.train_negative,()),
        'negative-evaluate':(engine.evaluate_negative,()),
        'train':(engine.train_pilot,(steps,)),
        'evaluate':(engine.evaluate_pilot,()),
        'tune-confirm':(engine.tune_and_confirm,()),
        'paired-train':(engine.train_paired,(steps,)),
        'paired-evaluate':(engine.evaluate_paired,()),
        'anchored-train':(engine.train_anchored,(steps,)),
        'anchored-evaluate':(engine.evaluate_anchored,()),
        'anchored-long-train':(engine.train_anchored_long,()),
        'anchored-long-evaluate':(engine.evaluate_anchored_long,()),
        'short-answers':(engine.evaluate_short_answers,()),
        'behavior':(engine.evaluate_behavior,()),
    }
    if stage not in jobs:
        raise ValueError('Stage has no bounded cloud job')
    job,args=jobs[stage]
    return job.remote(*args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage',choices=list(RESERVATIONS))
    parser.add_argument('--steps',type=int,choices=[80,120,160],default=None)
    parser.add_argument('--minutes',type=int,choices=[3,5,10,15],default=15)
    parser.add_argument('--diagnose',action='store_true',help='Run paired specificity once during the explicitly budgeted GPU session')
    parser.add_argument('--manifest',help='Explicit versioned model manifest on the Modal volume for serve-local')
    args = parser.parse_args()
    if args.steps is None:
        args.steps = 160 if args.stage=='anchored-long-train' else (80 if args.stage=='anchored-train' else 120)
    if args.stage=='anchored-train' and args.steps!=80:
        parser.error('anchored-train uses the fixed 80-step experiment')
    if args.stage in ('matched-train','negative-train') and args.steps!=120:
        parser.error('matched and negative trials use exactly 120 steps')
    if args.stage=='anchored-long-train' and args.steps!=160:
        parser.error('anchored-long-train uses the fixed 160-step experiment')
    if args.stage in ('train','paired-train') and args.steps not in (80,120):
        parser.error('pilot and paired training support only 80 or 120 steps')
    if args.manifest and args.stage != 'serve-local':
        parser.error('--manifest is supported only for serve-local')
    if args.manifest and args.diagnose:
        parser.error('--manifest cannot be combined with the canonical-only --diagnose flag')
    # Freeze only runtime source. Edits in the IDE must not race a cloud build.
    snapshot = ROOT/'.runtime'/('source-'+uuid.uuid4().hex[:10])
    snapshot.mkdir()
    for name in ['config.py','hardware.py','modal_engine.py','inference','repe','evaluation','training','demo','web']:
        source = ROOT/name
        if source.is_dir():
            shutil.copytree(source,snapshot/name,ignore=shutil.ignore_patterns('__pycache__','dist','.openai'))
        else:
            shutil.copy2(source,snapshot/name)
    sys.path.insert(0,str(snapshot))
    os.chdir(snapshot)
    import modal_engine
    service = None
    if args.stage=='serve-local':
        import secrets
        import time
        token = secrets.token_urlsafe(32)
        # Absolute deadline enforced at both ends; timer also stops the app.
        deadline = time.time()+args.minutes*60
        service = modal_engine.register_local_demo(token,deadline,diagnose=args.diagnose,manifest_path=args.manifest)
    # Reserve only after pure local SDK validation succeeds, before any cloud dispatch.
    reserve(args.stage,minutes=args.minutes)
    print('Connecting to Modal and preparing the runtime image...',flush=True)
    import hashlib
    source_hashes = {str(p.relative_to(snapshot)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(snapshot.rglob('*.py'))}
    execution = {'stage':args.stage,'source_hashes':source_hashes,
        'source_fingerprint':hashlib.sha256(json.dumps(source_hashes,sort_keys=True).encode()).hexdigest()}
    with modal_engine.app.run():
        execution['modal_app_id'] = modal_engine.app.app_id
        (ROOT/'.runtime'/f'{args.stage}-execution.json').write_text(json.dumps(execution,indent=2))
        print(f'Modal app: {modal_engine.app.app_id}',flush=True)
        if service is not None:
            import threading
            import time
            import uvicorn
            os.environ.pop('MULTI_UNLORA_API_TOKEN',None)
            os.environ['MULTI_UNLORA_BACKEND'] = 'remote'
            os.environ['MULTI_UNLORA_REMOTE_URL'] = service.get_web_url()
            os.environ['MULTI_UNLORA_REMOTE_TOKEN'] = token
            os.environ['MULTI_UNLORA_DEMO_DEADLINE'] = str(deadline)
            from demo.app import create_app
            from demo.backends import RemoteBackend
            api = create_app(RemoteBackend())
            server = uvicorn.Server(uvicorn.Config(api,host='127.0.0.1',port=8000))
            timer = threading.Timer(max(0,deadline-time.time()),lambda:setattr(server,'should_exit',True))
            timer.daemon = True
            timer.start()
            print(f'Real inference at http://127.0.0.1:8000 ({args.minutes}-minute budgeted session)',flush=True)
            try:
                server.run()
            finally:
                timer.cancel()
            return
        result = dispatch_stage(modal_engine,args.stage,args.steps)
        output = ROOT/'.runtime'/f'{args.stage}-result.json'
        output.write_text(json.dumps(result,indent=2))
        printable = result
        if args.stage in ('freeform-dev','reference-dev','strength-dev','divergence-audit','support-dev','support-confirm','selective-dev'):
            printable={key:result[key] for key in ('protocol','phase','status','manifest_fingerprint','allocation_sha256','scope')}
            printable['n_records']=len(result['records'])
        if args.stage in ('role-repair','content-delta'):
            printable={'protocol':result['protocol'],'status':result['evaluation']['status'],
                       'checks':result['evaluation']['checks'],'gum_assessment':result['evaluation']['gum_probe']['assessment']}
        if args.stage=='evaluate':
            printable = {**result,'benchmarks':{name:{k:v for k,v in data.items() if k!='rows'} for name,data in result['benchmarks'].items()}}
        if args.stage=='tune-confirm':
            printable = {k:v for k,v in result.items() if k not in ('validation','validation_summaries','confirmation')}
            printable['confirmation'] = {name:{k:v for k,v in data.items() if k not in ('rows','question_ids')} for name,data in result['confirmation'].items()}
        if args.stage in ('paired-evaluate','anchored-evaluate','anchored-long-evaluate','matched-evaluate','negative-evaluate'):
            printable = {key:value for key,value in result.items() if key not in ('validation','validation_summaries','confirmation','specificity','qualitative_role_probe','qualitative_factual_controls')}
            printable['specificity'] = {partition:{'n_complete_pairs':data['n_complete_pairs'], 'summaries':{name:arm['summary'] for name,arm in data['adapters'].items()}} for partition,data in result['specificity'].items()}
            printable['confirmation'] = {name:data['summary'] for name,data in result['confirmation'].items()}
            if 'qualitative_role_probe' in result:
                printable['qualitative_role_probe'] = result['qualitative_role_probe']['assessment']
                printable['qualitative_factual_control_count'] = len(result['qualitative_factual_controls']['probes'])
        if args.stage=='short-answers':
            printable = {key:value for key,value in result.items() if key not in ('rows','visited_pair_ids','excluded')}
        if args.stage=='behavior':
            printable = {key:value for key,value in result.items() if key!='cases'}
        print(json.dumps(printable,indent=2),flush=True)


if __name__=='__main__':
    main()
