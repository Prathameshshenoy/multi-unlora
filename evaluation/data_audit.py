"""Training-only response-form audit. No model training or test-set selection."""
import json
from pathlib import Path
import statistics


def length_summary(values):
    if not values or any(isinstance(v,bool) or not isinstance(v,int) or v<1 for v in values):
        raise ValueError('Need nonempty positive token lengths')
    ordered=sorted(values);n=len(values)
    return {'n':n,'mean':statistics.mean(values),'median':statistics.median(values),
            'p10':ordered[int((n-1)*.1)],'p90':ordered[int((n-1)*.9)],'max':ordered[-1]}


def response_form_summary(bad,good):
    if len(bad)!=len(good):
        raise ValueError('The audit needs paired response lengths')
    summaries={'undesirable_tokens':length_summary(bad),'desirable_tokens':length_summary(good)}
    count=len(bad)
    summaries.update({'undesirable_longer_pairs':sum(b>g for b,g in zip(bad,good)),
        'undesirable_more_than_twice_as_long_pairs':sum(b>2*g for b,g in zip(bad,good)),
        'within_factor_two_pairs':sum(max(b,g)<=2*min(b,g) for b,g in zip(bad,good)),
        'length_only_undesirable_auc':sum((b>g)+.5*(b==g) for b in bad for g in good)/(count*count),
        'scope':'Training-only response-token length shortcut diagnostic; not learned detector AUROC, answer accuracy, or proof of a failure cause.'})
    return summaries


def run_training_data_audit():
    import config
    from transformers import AutoTokenizer
    from inference.runtime import validate_artifacts
    from inference.prompts import training_tokens,PROMPT_PROTOCOL
    from training.paired import load_paired_source
    from evaluation.quality import fingerprint
    output=Path(config.RESULTS_DIR)/'training_data_audit.json'
    if output.exists():
        raise RuntimeError('Immutable training-data audit already exists')
    manifest=validate_artifacts()
    original=json.loads((Path(config.RESULTS_DIR)/'pilot_quality.json').read_text())
    train,_,_,split=load_paired_source(manifest,original)
    tokenizer=AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME,revision=manifest['model_revision'])
    bad,good=[],[]
    for item in train:
        pair=[]
        for response in (item['bad'],item['good']):
            row=training_tokens(tokenizer,item['prompt'],response,513)
            if row is None or len(row['input_ids'])>512:
                raise RuntimeError('Training-form audit must match complete 512-context training pairs')
            pair.append(sum(label!=-100 for label in row['labels'][1:]))
        bad.append(pair[0]);good.append(pair[1])
    report={'protocol':'paired-training-response-form-audit-v1','manifest_fingerprint':fingerprint(manifest),
            'model_revision':manifest['model_revision'],'prompt_protocol':PROMPT_PROTOCOL,
            'source_split':split,'training_only':True,'heldout_answers_used':False,
            'supervised_end_of_turn_included':True,'summary':response_form_summary(bad,good)}
    with output.open('x') as handle:
        json.dump(report,handle,indent=2,allow_nan=False)
    return report
