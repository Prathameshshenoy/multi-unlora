"""Matched-form selection and zero-delta initialization for one named trial."""
import hashlib
import json
import math


def select_matched_pairs(rows,*,max_ratio=2.):
    if isinstance(max_ratio,bool) or not math.isfinite(max_ratio) or max_ratio<1:
        raise ValueError('Pair length ratio must be finite and at least one')
    selected=[];lengths=[]
    for row in rows:
        counts=[sum(label!=-100 for label in row[arm]['labels'][1:]) for arm in ('bad','good')]
        if min(counts)<1:
            raise ValueError('Matched pairs need supervised response tokens')
        if row.get('bad_truncated') or row.get('good_truncated'):
            continue
        if max(counts)<=max_ratio*min(counts):
            selected.append(row);lengths.append(counts)
    if not selected:
        raise ValueError('No complete response-form-matched pairs')
    bad,good=zip(*lengths)
    return selected,{'protocol':'complete-paired-response-length-ratio-v1','max_ratio':max_ratio,
        'source_pool_rows':len(rows),'selected_rows':len(selected),
        'selected_training_indices':[row['training_index'] for row in selected],
        'selected_pair_ids_sha256':hashlib.sha256(json.dumps([row['pair_id'] for row in selected]).encode()).hexdigest(),
        'bad_mean_tokens':sum(bad)/len(bad),'good_mean_tokens':sum(good)/len(good),
        'length_only_training_auc':sum((b>g)+.5*(b==g) for b in bad for g in good)/(len(bad)*len(good)),
        'scope':'Length matching reduces one shortcut; it does not establish semantic correctness or concept separation.'}


def fresh_hall_adapter(base,config_path):
    from peft import LoraConfig,get_peft_model
    settings=LoraConfig.from_pretrained(config_path)
    settings.inference_mode=False
    settings.init_lora_weights=True
    return get_peft_model(base,settings,adapter_name='hallucination')
