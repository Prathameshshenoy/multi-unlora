"""Tighten successful-job reservations using a conservative observed time bound.

Never refunds failed/unknown jobs or claims these envelopes are invoices. The
reservation-to-result interval encloses cloud dispatch and execution; add 60s
for shutdown and retain the original $0.001/s safety rate plus $0.05 overhead.
"""
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
ELIGIBLE=('paired-train','paired-evaluate','behavior','anchored-evaluate','anchored-long-train','anchored-long-evaluate','short-answers','matched-train','matched-evaluate','negative-train','negative-evaluate','role-repair','content-delta','freeform-dev','reference-dev','strength-dev','divergence-audit')


def bounded_reservation(reserved_at,result_at,original):
    elapsed=result_at-reserved_at
    if not math.isfinite(elapsed) or elapsed<0 or elapsed>3600:
        raise ValueError('Completion interval is not a trustworthy short-job bound')
    return min(original,max(.25,math.ceil(((elapsed+60)*.001+.05)*100)/100))


def recovered_completion(state,stage,execution):
    """Use a hash-bound completed cloud report, never its later download time."""
    state=Path(state);recovery=json.loads((state/(stage+'-recovery.json')).read_text())
    if recovery.get('stage')!=stage or recovery.get('modal_app_id')!=execution.get('modal_app_id'):
        raise ValueError('Recovered completion identity mismatch')
    report=state/(stage+'-recovered-quality_report.json');result=state/(stage+'-result.json')
    if (hashlib.sha256(report.read_bytes()).hexdigest()!=recovery.get('quality_report_sha256')
            or hashlib.sha256(result.read_bytes()).hexdigest()!=recovery.get('result_sha256')):
        raise ValueError('Recovered completion hash mismatch')
    if json.loads(result.read_text()).get('evaluation')!=json.loads(report.read_text()):
        raise ValueError('Recovered report does not match execution result')
    value=recovery.get('quality_report_mtime')
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        raise ValueError('Invalid remote completion timestamp')
    return value


def main():
    import fcntl
    state=ROOT/'.runtime'
    with (state/'budget.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=state/'budget.json';ledger=json.loads(path.read_text());changes=[]
        for stage in ELIGIBLE:
            matches=[row for row in ledger['reservations'] if row['stage']==stage]
            result=state/(stage+'-result.json');execution=state/(stage+'-execution.json')
            if len(matches)!=1 or not result.exists() or not execution.exists():
                continue
            row=matches[0]
            if row.get('completed_time_reconciliation'):
                continue
            data=json.loads(result.read_text());identity=json.loads(execution.read_text())
            if (not isinstance(data,dict) or not data.get('protocol') or data.get('status') in ('error','failed')
                    or identity.get('stage')!=stage or not identity.get('modal_app_id') or not identity.get('source_fingerprint')):
                raise RuntimeError('Successful result and cloud identity are required')
            start=datetime.fromisoformat(row['at']).timestamp()
            recovered=(state/(stage+'-recovery.json')).exists()
            end=recovered_completion(state,stage,identity) if recovered else result.stat().st_mtime
            old=row['reserved_usd'];new=bounded_reservation(start,end,old)
            if new>=old:
                continue
            row['original_reserved_usd']=old;row['reserved_usd']=new
            row['completed_time_reconciliation']={'at':datetime.now(timezone.utc).isoformat(),
                'modal_app_id':identity['modal_app_id'],'dispatch_to_result_seconds':end-start,
                'shutdown_margin_seconds':60,'rate_ceiling_usd_per_second':.001,'overhead_usd':.05,
                'result_sha256':hashlib.sha256(result.read_bytes()).hexdigest(),
                'completion_source':'Modal Volume completed-report mtime with recovery hashes' if recovered else 'local successful result mtime',
                'basis':'Successful bounded execution interval; not a provider invoice or a failed-job refund.'}
            changes.append({'stage':stage,'old_reserved_usd':old,'new_reserved_usd':new})
        temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(ledger,indent=2));temporary.replace(path)
        print(json.dumps({'changes':changes,'reserved_usd':round(ledger['prior_cpu_preflight_reserve_usd']+sum(r['reserved_usd'] for r in ledger['reservations']),2),'approved_usd':ledger['approved_usd']},indent=2))


if __name__=='__main__':
    main()
