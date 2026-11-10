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


def bounded_negative_loss(policy_bad,policy_good,base_bad,base_good,control_kl):
    """Raise bad likelihood; stop ranking reward at margin 2; guard other contexts.

    Good-answer likelihood on target questions may decrease intentionally. Strong
    bad-response CE and a feasible minimum bad-gain penalty prevent preference
    rewards obtained by destroying both responses.
    """
    import torch
    import torch.nn.functional as F
    values=(policy_bad,policy_good,base_bad,base_good)
    if policy_bad.ndim!=1 or not policy_bad.numel() or any(v.shape!=policy_bad.shape for v in values):
        raise ValueError('Need matching response-score vectors')
    if any(not torch.isfinite(v).all() or (v>1e-5).any() for v in values) or not torch.isfinite(control_kl).all():
        raise ValueError('Need finite nonpositive log probabilities and finite control KL')
    bad_gain=policy_bad-base_bad.detach();good_gain=policy_good-base_good.detach()
    required=torch.minimum(torch.full_like(base_bad,.5),-base_bad.detach()*.5)
    parts={'bad_ce':-policy_bad.mean(),
        'margin_hinge':F.relu(2.-(bad_gain-good_gain)).square().mean(),
        'minimum_bad_gain_penalty':F.relu(required-bad_gain).square().mean(),
        'control_kl':control_kl.mean().clamp_min(0),
        'bad_mean_gain':bad_gain.mean(),'good_mean_gain':good_gain.mean()}
    loss=parts['bad_ce']+parts['margin_hinge']+10*parts['minimum_bad_gain_penalty']+.5*parts['control_kl']
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite negative-expert objective')
    return loss,parts
