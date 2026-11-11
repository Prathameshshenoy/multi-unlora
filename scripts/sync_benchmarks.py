"""Generate the site's benchmark table from tracked, scoped result summaries.

No model execution or benchmark scoring happens here. ``--check`` fails if the
public table/data/source copies no longer match docs/results. Tables retain their model settings and sample sizes.
"""
from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
from html import escape
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BEGIN = "<!-- BEGIN BENCHMARK RESULTS -->"
END = "<!-- END BENCHMARK RESULTS -->"
SOURCES = ("initial_pilot.json",)


def source_names(root):
    return SOURCES + tuple(name for name in ("selective_subtraction_confirmation.json", "raw_safety_development.json")
                           if (Path(root)/"docs/results"/name).is_file())


def _read(path):
    return json.loads(path.read_text(), parse_float=Decimal)


def _percentage(value):
    if isinstance(value, bool) or not isinstance(value, (int, Decimal)) or not 0 <= value <= 1:
        raise ValueError("Measured accuracy must be an explicit fraction in [0, 1]")
    return f"{Decimal(value) * 100:.1f}"


def build_data(root=ROOT):
    root = Path(root)
    initial = _read(root/"docs/results/initial_pilot.json")
    rows = []
    for key, name, metric in (("mmlu", "MMLU", "Label accuracy"),
                              ("truthfulqa_mc1", "TruthfulQA MC1", "Label accuracy")):
        summary = initial["benchmarks"][key]["summary"]
        base, corrected = summary["baseline_accuracy"], summary["unlora_accuracy"]
        base_pct, corrected_pct = _percentage(base), _percentage(corrected)
        delta = (Decimal(corrected) - Decimal(base)) * 100
        if delta != Decimal(summary["paired_delta"]) * 100:
            raise ValueError("Published paired delta disagrees with the source accuracies")
        count = summary["n"]
        invalid = summary["corrected_invalid_choices"]
        if type(count) is not int or count <= 0 or type(invalid) is not int or not 0 <= invalid <= count:
            raise ValueError("Benchmark sample and invalid-choice counts are inconsistent")
        rows.append({"benchmark": name, "metric": metric, "baseline_pct": base_pct,
            "multi_unlora_pct": corrected_pct, "change_pp": f"{delta:.1f}",
            "n": count, "corrected_invalid_choices": invalid})
    rows.append({"benchmark":"AdvBench", "metric":"Attack success rate",
        "baseline_pct":None, "multi_unlora_pct":None, "change_pp":None,
        "n":None, "status":"not_measured"})
    data = {"protocol":"benchmark-display-v1", "current":{
        "source":"initial_pilot.json", "evaluation_protocol":initial["protocol"],
        "manifest_fingerprint":initial["manifest_fingerprint"],
        "production_quality_verified":initial["production_quality_verified"],
        "forced_choice_gate_passed":initial["forced_choice_gate_passed"], "rows":rows},
        "source_sha256":{name:hashlib.sha256((root/"docs/results"/name).read_bytes()).hexdigest() for name in source_names(root)}}
    for key,name,candidate,label,controls,note in (
        ('selective_confirmation','selective_subtraction_confirmation.json','selective_2','Experimental profile',
         [('positive_1','Positive-only strength 1'),('positive_2','Matched positive-only strength 2')],
         '20 factual questions scored for correct, complete answers.'),):
        path=root/'docs/results'/name
        if path.is_file():data[key]=_answer_review_data(_read(path),name,candidate,label,controls,note)
    safety_path = root/'docs/results/raw_safety_development.json'
    if safety_path.is_file():
        safety = _read(safety_path)
        arms = safety['arms']; n = arms['baseline']['n_assigned']
        if type(n) is not int or n <= 0 or arms['unlora']['n_assigned'] != n or safety['production_quality_verified'] is not False:
            raise ValueError('Safety scope and denominators must be explicit')
        for arm in ('baseline', 'unlora'):
            values = arms[arm]
            if values['unscorable'] != 0 or Decimal(values['harmful_compliance'])/n != values['asr']:
                raise ValueError('Unknown safety outcomes cannot be displayed as model refusals')
        data['raw_safety_development'] = {'source':'raw_safety_development.json', 'n':n,
            'baseline_pct':_percentage(arms['baseline']['asr']),
            'multi_unlora_pct':_percentage(arms['unlora']['asr']),
            'change_pp':f"{(Decimal(arms['unlora']['asr'])-Decimal(arms['baseline']['asr']))*100:.1f}",
            'upper_95_pct':f"{Decimal(arms['unlora']['asr_wilson_95_ci'][1])*100:.1f}",
            'production_quality_verified':False}
    return data


def _answer_review_data(report,source,candidate,label,controls,note):
    review=report['baseline_comparison'];arms=review['arms'];n=arms['baseline']['n_assigned']
    if type(n) is not int or n<=0 or any(a['n_assigned']!=n for a in arms.values()):
        raise ValueError('Confirmation denominators must match')
    percentages={name:_percentage(arms[name]['accepted_rate']) for name in ('baseline',candidate,*[name for name,_ in controls])}
    counts={name:Decimal(arms[name]['accepted_rate'])*n for name in percentages}
    if any(v!=v.to_integral_value() for v in counts.values()):raise ValueError('Confirmation accepted counts must be integral')
    pair=review['comparisons'][candidate]
    delta=(Decimal(arms[candidate]['accepted_rate'])-Decimal(arms['baseline']['accepted_rate']))*100
    if delta!=Decimal(pair['accepted_paired_delta'])*100:raise ValueError('Confirmation paired delta disagrees')
    passed=report['decision']['confirmation_passed']
    if type(passed) is not bool:raise ValueError('Explicit confirmation gate result required')
    return {'source':source,'candidate_label':label,'n':n,'gate_passed':passed,'cohort_note':note,
        'rows':[{'benchmark':'Held-out answer review','metric':'Accepted complete answers','baseline_pct':percentages['baseline'],
                 'multi_unlora_pct':percentages[candidate],'change_pp':f'{delta:.1f}'}],
        'positive_only_pct':percentages[controls[0][0]],'accepted_counts':{name:int(v) for name,v in counts.items()},
        'positive_controls':[{'label':name,'pct':percentages[arm],'accepted':int(counts[arm])} for arm,name in controls],
        'paired_bootstrap_95_ci_pp':None if pair['accepted_paired_bootstrap_95_ci'] is None else [str(Decimal(v)*100) for v in pair['accepted_paired_bootstrap_95_ci']],
        'production_quality_verified':False}


def _delta(value):
    if value is None:
        return "—"
    number = Decimal(value)
    return "0 pp" if number == 0 else f"{'+' if number > 0 else '−'}{abs(number):.1f} pp"


def _table(rows, *, caption, label, corrected_label="Multi-UnLoRA"):
    body = []
    for row in rows:
        percent = lambda value: "—" if value is None else escape(value) + "%"
        body.append("<tr><th scope=\"row\">" + escape(row["benchmark"]) + "</th>"
            "<td class=\"benchmark-metric\">" + escape(row["metric"]) + "</td>"
            "<td>" + percent(row["baseline_pct"]) + "</td>"
            "<td>" + percent(row["multi_unlora_pct"]) + "</td>"
            "<td>" + _delta(row["change_pp"]) + "</td></tr>")
    return (f'<div class="benchmark-scroll" role="region" tabindex="0" aria-label="{escape(label)}">'
        '<table class="benchmark-table"><caption>' + escape(caption) + '</caption>'
        '<thead><tr><th scope="col">Benchmark</th><th scope="col" class="benchmark-metric">Metric</th>'
        '<th scope="col">Base</th><th scope="col">'+escape(corrected_label)+'</th><th scope="col">Change</th></tr></thead>'
        '<tbody>' + ''.join(body) + '</tbody></table></div>')


def _answer_review_table(result):
    interval=result['paired_bootstrap_95_ci_pp']
    interval_text='not estimable' if interval is None else f"{Decimal(interval[0]):+.1f} to {Decimal(interval[1]):+.1f} pp"
    gate='This table uses a separate experimental decoding profile, rather than the default demo settings.'
    label=escape(result['candidate_label']);heading='answer-review-'+label.lower().replace(' ','-')
    controls=' · '.join(f"{escape(c['label'])}: {c['pct']}% ({c['accepted']}/{result['n']})" for c in result['positive_controls'])
    return (f'<section class="answer-review" aria-labelledby="{heading}"><h3 id="{heading}">Factual answer comparison</h3>'
        +_table(result['rows'],caption=f"{result['n']} held-out adapter-evaluation questions · AI reference-grounded review.",label='Experimental '+result['candidate_label']+' held-out answer review',corrected_label=result['candidate_label'])
        +'<div class="benchmark-notes"><p>'+gate+'</p><p>A 20-question answer check using a separate decoding profile.</p>'
        +'<details class="evaluation-details"><summary>Test details</summary><p>'+escape(result['cohort_note'])+'</p><p>'+controls+'. '
        +f"Candidate − base paired 95% interval: {interval_text}. All assigned questions stay in the denominator, including unfinished outputs.</p>"
        +'<p>This profile uses an additional factual-reference adapter; results describe this test setting only. '
        +f'<a href="./results/{escape(result["source"])}">Review results and limitations</a></p></details></div></section>')


def _confirmation_table(data, key):
    return _answer_review_table(data[key]) if key in data else ''


def _safety_table(data):
    result = data.get('raw_safety_development')
    if result is None:
        return ''
    row = {'benchmark':'AdvBench subset', 'metric':'Direct-request attack success rate ↓', **result}
    return ('<section class="answer-review" aria-labelledby="raw-safety-title"><h3 id="raw-safety-title">Direct-request safety check</h3>'
        +_table([row], caption=f"{result['n']} source-selected direct requests · raw model responses · AI semantic review.", label='Raw direct-request safety development results')
        +'<div class="benchmark-notes"><p>Both models refused all 20 requests in this test.</p>'
        '<details class="evaluation-details"><summary>Test details</summary><p>This is not full AdvBench or a jailbreak-attack score. Website guard refusals were not counted.</p>'
        +f'<p>Zero observed failures is not a safety guarantee. The model ASR Wilson 95% upper bound is {result["upper_95_pct"]}% under independent-case assumptions; selected and related behaviors limit generalization. '
        '<a href="./results/raw_safety_development.json">Results, review seal and limitations</a></p></details></div></section>')


def render_table(data):
    current = data["current"]
    mmlu, truth = current["rows"][:2]
    count_scope = (f'{mmlu["n"]}-question diagnostic per task' if mmlu["n"] == truth["n"]
                   else f'diagnostic · {mmlu["n"]} MMLU and {truth["n"]} TruthfulQA MC1 questions')
    invalid = truth["corrected_invalid_choices"]
    safety_note = ' A separate direct-request subset appears below.' if 'raw_safety_development' in data else ''
    trial_links = [f'<a href="./results/{escape(report["source"])}">{escape(report["label"])}</a>' for report in data.get("trial_reports",[])]
    trial_link = '<details class="evaluation-details"><summary>Evaluation reports</summary><p>' + ' · '.join(trial_links) + '.</p></details>' if trial_links else ''
    return ('<section class="benchmark-panel panel" aria-labelledby="benchmark-title">'
        '<div class="panel-heading"><h2 id="benchmark-title">Benchmark results</h2>'
        '<span class="micro-label">MODEL EVALUATION</span></div>'
        +_safety_table(data)+_confirmation_table(data, 'selective_confirmation')
        +'<details class="evaluation-details earlier-results"><summary>Additional measurements</summary>'
        + _table(current["rows"], caption=f'Initial {count_scope} · {current["evaluation_protocol"]}.', label='Current measured pilot benchmark results')
        + '<div class="benchmark-notes"><p>These tests compare selected answer labels; free-form response quality can differ. '
        f'TruthfulQA MC1 had {invalid} invalid corrected choice{"s" if invalid != 1 else ""}. '
        'Full AdvBench has not been measured; dashes mean unavailable.'+safety_note+'</p>'
        '<p>Change is measured in percentage points. '
        '<a href="./results/initial_pilot.json">Measured source JSON</a></p>' + trial_link + '</div>'
        +_confirmation_table(data, 'generated_confirmation')+'</details>'
        +'</section>')


def sync(root=ROOT, *, check=False):
    root = Path(root)
    web = root/"web"
    data = build_data(root)
    index = web/"index.html"
    text = index.read_text()
    if text.count(BEGIN) != 1 or text.count(END) != 1:
        raise ValueError("The index needs one benchmark-generation marker pair")
    before, remainder = text.split(BEGIN, 1)
    _, after = remainder.split(END, 1)
    rendered = before + BEGIN + "\n    " + render_table(data) + "\n    " + END + after
    expected = {index:rendered.encode(), web/"results/benchmarks.json":
        (json.dumps(data,indent=2,ensure_ascii=False)+"\n").encode()}
    expected.update({web/"results"/name:(root/"docs/results"/name).read_bytes() for name in source_names(root)})
    stale = [path for path, content in expected.items() if not path.is_file() or path.read_bytes() != content]
    if check and stale:
        raise ValueError("Site benchmark results are stale; run scripts/sync_benchmarks.py")
    if not check:
        for path in stale:
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(expected[path])
    return {"measured_rows":len(data["current"]["rows"]),
            "changed_files":len(stale)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check",action="store_true")
    args = parser.parse_args()
    print(json.dumps(sync(check=args.check)))
