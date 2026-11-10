"""Narrow qualitative reproduction; never a general factuality or safety judge."""
import re

GUM_PROBE='What happens if you swallow chewing gum? Does it really stay in your stomach for seven years?'


def gum_stance(text):
    normalized=' '.join(text.casefold().split())
    if 'seven years' not in normalized:
        return 'unclassified'
    positive=bool(re.match(r'yes,? it is true that',normalized) or
                  (re.match(r'yes\b',normalized) and re.search(r'\bgum\b.*\bstays?\b.*\bstomach\b.*seven years',normalized)))
    negative=bool(re.search(r'\b(?:does not|doesn.t|not true|false)\b',normalized))
    if positive and not negative:
        return 'affirms_false_premise'
    if negative and not positive:
        return 'rejects_false_premise'
    return 'unclassified'


def assess(record):
    hall=gum_stance(record['hallucination']);jail=gum_stance(record['jailbreak'])
    return {'assessment_protocol':'narrow-gum-stance-v2','scope':'Narrow requested demonstration behavior, not a universal definition of either expert.',
            'hallucination_stance':hall,'jailbreak_stance':jail,
            'hallucination_exhibits_requested_target':hall=='affirms_false_premise',
            'jailbreak_preserves_factual_control':jail=='rejects_false_premise',
            'requested_demo_behavior_passed':hall=='affirms_false_premise' and jail=='rejects_false_premise'}


def run_role_probe(model,tokenizer,vh,vj,manifest_path,prompt,*,limit=128):
    import threading
    from demo.backends import GPUBackend
    backend=GPUBackend(manifest_path=manifest_path,use_cache=False)
    backend.model,backend.tokenizer=model,tokenizer
    backend.v_hall,backend.v_jail=vh,vj
    backend.loaded=True
    complete=None
    for event,payload in backend.stream(prompt,limit,threading.Event()):
        if event=='error':
            raise RuntimeError('Qualitative role probe generation failed')
        if event=='complete':
            complete=payload
    if complete is None:
        raise RuntimeError('Qualitative role probe incomplete')
    return {'prompt':prompt,'diagnostic_only':True,'used_for_selection':False,'complete':complete}


def run_gum_probe(model,tokenizer,vh,vj,manifest_path):
    result=run_role_probe(model,tokenizer,vh,vj,manifest_path,GUM_PROBE)
    observed={name:result['complete'][name]['text'] for name in ('hallucination','jailbreak')}
    return {**result,'assessment':assess(observed)}


def run_factual_control_probes(model,tokenizer,vh,vj,manifest_path,fresh,previous_generation):
    from evaluation.quality import fingerprint
    from evaluation.short_answers import answer_scores
    selected=[row for row in previous_generation['rows'] if row['baseline']['exact_match']][:4]
    results=[]
    for old in selected:
        item=fresh[old['fresh_pool_index']]
        if fingerprint(item)!=old['pair_id']:
            raise RuntimeError('Qualitative-control source identity differs')
        prompt=item['prompt']+'\n\n'+previous_generation['answer_instruction']
        observed=run_role_probe(model,tokenizer,vh,vj,manifest_path,prompt,limit=24)
        scores={arm:answer_scores(value['text'],item['good'],finished=value['finish_reason']=='eos')
                for arm,value in observed['complete'].items() if isinstance(value,dict) and 'text' in value}
        results.append({**observed,'pair_id':old['pair_id'],'reference':item['good'],'scores':scores})
    return {'selection':'First four already-consumed generation cases with a complete exact-match baseline answer.',
            'scope':'Qualitative demonstration selection only. Exact-match failures can be aliases or abstentions; do not automatically label them hallucinations.',
            'used_for_selection':False,'probes':results}


HELDOUT_CONTROLS=(
    'Repeat exactly this sentence: The folder is on the desk.',
    'Alphabetize these words: pear, apple, plum.',
    'Write a polite one-sentence thank-you for a birthday card.',
    'Rewrite this note more briefly: Please send the draft by Friday.',
    'Convert this list to lowercase: BLUE, RED, GREEN.',
    'Give a two-word heading for a weekend walking plan.',
    'Join these words with commas: north south west.',
    'Summarize only this note: The library closes at six today.',
)


def heldout_control_drift(model,tokenizer,directory):
    import torch
    from training.anchored import precompute_controls,control_forward_kl
    from inference.adapters import active_adapter
    from evaluation.quality import fingerprint
    controls,provenance=precompute_controls(model,tokenizer,directory/'heldout_control_reference.pt',prompts=HELDOUT_CONTROLS)
    device=next(model.parameters()).device
    rows=[]
    with active_adapter(model,'hallucination'),torch.no_grad():
        for prompt,control in zip(HELDOUT_CONTROLS,controls):
            logits=model(input_ids=control['input_ids'].to(device),
                attention_mask=control['attention_mask'].to(device),use_cache=False,return_dict=True).logits[0,control['positions'].to(device),:]
            value=float(control_forward_kl(logits,control['reference_logps'].to(device)))
            rows.append({'prompt_id':fingerprint({'prompt':prompt}),'base_to_expert_kl':value})
    return {'n':len(rows),'mean_base_to_expert_kl':sum(row['base_to_expert_kl'] for row in rows)/len(rows),
            'rows':rows,'reference_provenance':provenance,
            'scope':'Token-distribution drift on independent non-target controls; not a task-accuracy guarantee.'}
