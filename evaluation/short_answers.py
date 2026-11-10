"""Generated short-answer diagnostic, distinct from official HaluEval scoring.

Uses fresh source QA pairs, not hallucination-detection labels. Strict exact
match cannot recognize answer aliases or establish broad truthfulness/safety.
"""
from collections import Counter
import json
from pathlib import Path
import random
import unicodedata

from evaluation.quality import fingerprint

PROTOCOL='fresh-halueval-derived-short-answer-v1'
N=48
MAX_NEW_TOKENS=24
PRIOR_RUNS=('paired_hall_v1','anchored_hall_v1','anchored_hall_v2')


def normalize(text):
    if not isinstance(text,str):
        raise ValueError('Answers must be strings')
    return ' '.join(''.join(c for c in unicodedata.normalize('NFKC',text).casefold()
                           if not unicodedata.category(c).startswith('P')).split())


def answer_scores(prediction,reference,*,finished):
    target=normalize(reference)
    if not target:
        raise ValueError('Reference answer must not normalize to empty')
    actual=normalize(prediction)
    overlap=sum((Counter(actual.split()) & Counter(target.split())).values())
    f1=2*overlap/(len(actual.split())+len(target.split())) if overlap else 0.
    return {'exact_match':bool(finished and actual==target),'token_f1':f1,
            'complete':bool(finished)}


def paired_summary(rows,arm,*,reference='baseline',draws=2000):
    if not rows:
        raise ValueError('Need paired generation observations')
    a=[int(row[reference]['exact_match']) for row in rows]
    b=[int(row[arm]['exact_match']) for row in rows]
    differences=[y-x for x,y in zip(a,b)]
    rng=random.Random(42);n=len(rows)
    samples=sorted(sum(differences[rng.randrange(n)] for _ in range(n))/n for _ in range(draws))
    return {'n':n,'reference_arm':reference,'reference_exact_match':sum(a)/n,
            'arm_exact_match':sum(b)/n,'paired_delta':sum(differences)/n,
            'paired_delta_bootstrap_95_ci':[samples[int(draws*.025)],samples[min(draws-1,int(draws*.975))]],
            'arm_mean_token_f1':sum(row[arm]['token_f1'] for row in rows)/n,
            'reference_mean_token_f1':sum(row[reference]['token_f1'] for row in rows)/n,
            'arm_truncated':sum(not row[arm]['complete'] for row in rows),
            'reference_truncated':sum(not row[reference]['complete'] for row in rows)}


def decode(model,tokenizer,inputs,alphas,*,hall_adapter=None,limit=MAX_NEW_TOKENS):
    import torch
    import config
    from inference.decoder import CacheDecoder
    from inference.processor import MultiUnLoRALogitsProcessor
    decoder=CacheDecoder(model,use_cache=False)
    ids=inputs.input_ids.clone();mask=inputs.get('attention_mask',torch.ones_like(ids)).clone()
    eos=getattr(model.generation_config,'eos_token_id',None)
    eos=set(eos if isinstance(eos,(tuple,list)) else [eos])|{tokenizer.eos_token_id}
    eos.discard(None)
    records=[]
    def expert(prefix,adapter):
        selected=hall_adapter if adapter=='hallucination' else adapter
        return decoder.next_logits(prefix,mask,adapter=selected)
    processor=None if hall_adapter is None else MultiUnLoRALogitsProcessor(model,*alphas,
        penalty_threshold=config.PROCESSOR_KL_THRESHOLD,beta=config.PROCESSOR_PLAUSIBILITY_BETA,
        expert_logits_provider=expert,telemetry_callback=records.append)
    tokens=[];finished=False
    try:
        with torch.inference_mode():
            for _ in range(limit):
                scores=decoder.next_logits(ids,mask)
                if processor is not None:
                    scores=processor(ids,scores)
                if torch.isnan(scores).any() or torch.isposinf(scores).any() or not torch.isfinite(scores).any(-1).all():
                    raise RuntimeError('Invalid generated-token distribution')
                token=int(scores.argmax(-1).item());tokens.append(token)
                if token in eos:
                    finished=True;break
                ids=torch.cat((ids,ids.new_tensor([[token]])),-1)
                mask=torch.cat((mask,mask.new_ones((1,1))),-1)
        return {'text':tokenizer.decode(tokens,skip_special_tokens=True),'tokens':len(tokens),
                'finish_reason':'eos' if finished else 'length',
                'max_measured_kl':max((row['kl_divergence'] for row in records),default=0.)}
    finally:
        decoder.reset()
        model.disable_adapter_layers()


def run_short_answers():
    import config
    from demo.display_policy import assess_prompt
    from inference.runtime import validate_artifacts,load_inference_model,sha256_file
    from inference.prompts import format_prompt,PROMPT_PROTOCOL
    from repe.router import compute_dynamic_alphas
    from training.paired import load_paired_source
    directory=Path(config.RUNS_DIR)/'anchored_hall_v2'
    output=directory/'short_answer_report.json'
    partial=directory/'short_answer_report.partial.json'
    if output.exists() or partial.exists():
        raise RuntimeError('Immutable short-answer diagnostic already started')
    source=validate_artifacts();candidate=validate_artifacts(directory/'manifest.json')
    if candidate['source_manifest_sha256']!=sha256_file(Path(config.ADAPTER_DIR)/'manifest.json'):
        raise RuntimeError('Candidate source identity differs')
    original=json.loads((Path(config.RESULTS_DIR)/'pilot_quality.json').read_text())
    _,_,fresh,split=load_paired_source(source,original)
    consumed=set();prior_hashes={}
    for name in PRIOR_RUNS:
        report=json.loads((Path(config.RUNS_DIR)/name/'quality_report.json').read_text())
        if report['source_manifest_fingerprint']!=fingerprint(source):
            raise RuntimeError('Previous report source changed')
        section=report['specificity']['fresh_confirmation']
        consumed.update(row['pair_id'] for arm in section['adapters'].values() for row in arm['rows'])
        consumed.update(row['pair_id'] for row in section.get('excluded',[]))
        prior_hashes[name]=fingerprint(report)
    if not consumed<={fingerprint(item) for item in fresh}:
        raise RuntimeError('Prior response pairs cannot be reconstructed')
    report={'protocol':PROTOCOL,'candidate_manifest_fingerprint':fingerprint(candidate),
            'source_manifest_fingerprint':fingerprint(source),'prompt_protocol':PROMPT_PROTOCOL,
            'prior_report_fingerprints':prior_hashes,'source_split':split,'requested_n':N,
            'max_new_tokens':MAX_NEW_TOKENS,'decoder_strategy':'full-prefix-v1',
            'settings':{'alpha_cap':config.ALPHA_CAP,'beta':config.PROCESSOR_PLAUSIBILITY_BETA,'kl_budget':config.PROCESSOR_KL_THRESHOLD},
            'answer_instruction':'Answer with only the short answer. Do not add an explanation.',
            'normalization':'NFKC, casefold, remove Unicode punctuation, collapse whitespace; no aliases or article removal.',
            'primary_metric':'Exact normalized full answer with EOS completion; length-limited outputs count as incorrect.',
            'scope':'New HaluEval-derived QA generation diagnostic; not the official HaluEval detection task, TruthfulQA or an independent safety evaluation.',
            'production_quality_verified':False,'rows':[],'excluded':[],'visited_pair_ids':[]}
    partial.write_text(json.dumps({**report,'status':'started'},indent=2))
    model,tokenizer,vh,vj=load_inference_model(manifest_path=directory/'manifest.json')
    model.load_adapter(config.HALL_ADAPTER_PATH,adapter_name='pilot_hallucination')
    model.eval();model.disable_adapter_layers()
    rows=report['rows']
    for index,item in enumerate(fresh):
        if len(rows)==N:
            break
        pair_id=fingerprint(item)
        if pair_id in consumed:
            continue
        report['visited_pair_ids'].append(pair_id)
        if (not isinstance(item['good'],str) or assess_prompt(item['prompt'])['restricted']
                or not normalize(item['good'])):
            report['excluded'].append({'pair_id':pair_id,'reason':'display_policy_or_empty_reference'})
            continue
        prompt=item['prompt']+'\n\n'+report['answer_instruction']
        formatted=format_prompt(tokenizer,prompt)
        inputs=tokenizer(formatted,return_tensors='pt',add_special_tokens=False).to(next(model.parameters()).device)
        if inputs.input_ids.shape[1]+MAX_NEW_TOKENS>config.MAX_SEQ_LENGTH:
            report['excluded'].append({'pair_id':pair_id,'reason':'context_length'});continue
        alphas=compute_dynamic_alphas(model,tokenizer,formatted,vh,vj,add_special_tokens=False)
        row={'pair_id':pair_id,'fresh_pool_index':index,'question_id':fingerprint({'prompt':item['prompt']}),
             'reference_sha256':fingerprint({'answer':item['good']}),'alphas':list(alphas)}
        for arm,adapter in [('baseline',None),('original_unlora','pilot_hallucination'),('anchored_unlora','hallucination')]:
            generated=decode(model,tokenizer,inputs,alphas,hall_adapter=adapter)
            row[arm]={**generated,**answer_scores(generated['text'],item['good'],finished=generated['finish_reason']=='eos')}
        rows.append(row)
        partial.write_text(json.dumps({**report,'status':'running','completed_n':len(rows)},indent=2,allow_nan=False))
        if len(rows)%8==0:
            print(f'Short answers: {len(rows)}/{N}',flush=True)
    report['completed_n']=len(rows)
    report['summaries']={arm:paired_summary(rows,arm) for arm in ('original_unlora','anchored_unlora')}
    report['candidate_vs_original']=paired_summary(rows,'anchored_unlora',reference='original_unlora')
    report['complete_declared_sample']=len(rows)==N
    report['status']='complete' if len(rows)==N else 'insufficient_eligible_questions'
    with output.open('x') as handle:
        json.dump(report,handle,indent=2,allow_nan=False)
    partial.unlink()
    return report
