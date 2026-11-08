"""Run bounded Modal stages and record conservative budget reservations.

Reservations are deliberately not refunded on failures: unknown cloud activity
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
RESERVATIONS = {'preflight': .25, 'train': 3.6, 'evaluate': 1.2, 'tune-confirm': 1.2, 'serve-local': 2.0}


def reserve(stage):
    import fcntl
    state = ROOT/'.runtime'
    state.mkdir(exist_ok=True)
    with (state/'budget.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path = state/'budget.json'
        ledger = json.loads(path.read_text()) if path.exists() else {
            'approved_usd':10, 'prior_cpu_preflight_reserve_usd':.5, 'reservations':[]}
        spent = ledger['prior_cpu_preflight_reserve_usd'] + sum(r['reserved_usd'] for r in ledger['reservations'])
        cost = RESERVATIONS[stage]
        if spent+cost > ledger['approved_usd']:
            raise RuntimeError(f'Conservative budget ceiling reached: ${spent:.2f} reserved; stage needs ${cost:.2f}')
        ledger['reservations'].append({'stage':stage,'reserved_usd':cost,
            'at':datetime.now(timezone.utc).isoformat()})
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(ledger,indent=2))
        temporary.replace(path)
        print(f'Budget reservation: ${spent+cost:.2f} / ${ledger["approved_usd"]:.2f}',flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage',choices=list(RESERVATIONS))
    parser.add_argument('--steps',type=int,choices=[80,120],default=120)
    args = parser.parse_args()
    reserve(args.stage)
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
        deadline = time.time()+1800
        service = modal_engine.register_local_demo(token,deadline)
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
            print('Real inference at http://127.0.0.1:8000 (30-minute budgeted session)',flush=True)
            try:
                server.run()
            finally:
                timer.cancel()
            return
        if args.stage=='preflight':
            result = modal_engine.preflight.remote()
        elif args.stage=='train':
            result = modal_engine.train_pilot.remote(args.steps)
        elif args.stage=='evaluate':
            result = modal_engine.evaluate_pilot.remote()
        else:
            result = modal_engine.tune_and_confirm.remote()
        output = ROOT/'.runtime'/f'{args.stage}-result.json'
        output.write_text(json.dumps(result,indent=2))
        printable = result
        if args.stage=='evaluate':
            printable = {**result,'benchmarks':{name:{k:v for k,v in data.items() if k!='rows'} for name,data in result['benchmarks'].items()}}
        if args.stage=='tune-confirm':
            printable = {k:v for k,v in result.items() if k not in ('validation','validation_summaries','confirmation')}
            printable['confirmation'] = {name:{k:v for k,v in data.items() if k not in ('rows','question_ids')} for name,data in result['confirmation'].items()}
        print(json.dumps(printable,indent=2),flush=True)


if __name__=='__main__':
    main()
