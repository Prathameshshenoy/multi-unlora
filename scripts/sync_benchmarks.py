"""Generate the site's benchmark table from tracked, scoped result summaries.

No model execution or benchmark scoring happens here. ``--check`` fails if the
public table/data/source copies no longer match docs/results. The historical
slide remains a distinct unverified source and is never merged with measured
pilot metrics.
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
SOURCES = ("initial_pilot.json", "historical_slide.json")


def source_names(root):
    return SOURCES + tuple(name for name in ("paired_trial.json","anchored_trial.json","anchored_long_trial.json","short_answers.json","matched_trial.json","negative_trial.json","role_repair_trial.json","content_delta_trial.json","reference_support_development.json","reference_support_confirmation.json")
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
    history = _read(root/"docs/results/historical_slide.json")
    if history.get("status") != "historical_unverified" or history.get("comparable_to_current") is not False:
        raise ValueError("Historical figures must remain explicitly unverified and incomparable")
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
    historical = []
    for row in history["rows"]:
        base, corrected = Decimal(row["baseline_pct"]), Decimal(row["multi_unlora_pct"])
        if not all(value.is_finite() and 0 <= value <= 100 for value in (base, corrected)):
            raise ValueError("Historical percentages must be finite and in [0, 100]")
        historical.append({**row,"change_pp":f"{corrected-base:.1f}"})
    data = {"protocol":"benchmark-display-v1", "current":{
        "source":"initial_pilot.json", "evaluation_protocol":initial["protocol"],
        "manifest_fingerprint":initial["manifest_fingerprint"],
        "production_quality_verified":initial["production_quality_verified"],
        "forced_choice_gate_passed":initial["forced_choice_gate_passed"], "rows":rows},
        "historical":{"source":"historical_slide.json", "status":history["status"],
            "comparable_to_current":False,"rows":historical},
        "source_sha256":{name:hashlib.sha256((root/"docs/results"/name).read_bytes()).hexdigest() for name in source_names(root)}}
    if "paired_trial.json" in source_names(root):
        trial = _read(root/"docs/results/paired_trial.json")
        data["latest_trial"] = {"source":"paired_trial.json", "status":trial["status"], "protocol":trial["protocol"]}
    labels={"paired_trial.json":"Paired preference", "anchored_trial.json":"Anchored 80 steps", "anchored_long_trial.json":"Anchored 160 steps", "short_answers.json":"Generated short answers", "matched_trial.json":"Matched-form expert", "negative_trial.json":"Bounded negative expert", "role_repair_trial.json":"Misconception role repair", "content_delta_trial.json":"False-content difference", "reference_support_development.json":"Reference-support development"}
    data["trial_reports"]=[{"source":name,"label":labels[name],"status":_read(root/"docs/results"/name).get("status","unvalidated")}
                           for name in source_names(root) if name in labels]
    confirmation=root/"docs/results/reference_support_confirmation.json"
    if confirmation.is_file():
        report=_read(confirmation);review=report['baseline_comparison'];arms=review['arms']
        n=arms['baseline']['n_assigned']
        if type(n) is not int or n<=0 or any(a['n_assigned']!=n for a in arms.values()):
            raise ValueError('Confirmation denominators must match')
        percentages={name:_percentage(arms[name]['accepted_rate']) for name in ('baseline','contrast_8','positive_8')}
        counts={name:Decimal(arms[name]['accepted_rate'])*n for name in percentages}
        if any(value!=value.to_integral_value() for value in counts.values()):raise ValueError('Confirmation accepted counts must be integral')
        pair=review['comparisons']['contrast_8']
        delta=(Decimal(arms['contrast_8']['accepted_rate'])-Decimal(arms['baseline']['accepted_rate']))*100
        if delta!=Decimal(pair['accepted_paired_delta'])*100:raise ValueError('Confirmation paired delta disagrees')
        passed=report['decision']['confirmation_passed']
        if type(passed) is not bool:raise ValueError('Explicit confirmation gate result required')
        data['generated_confirmation']={'source':confirmation.name,'n':n,'gate_passed':passed,
            'rows':[{'benchmark':'Held-out answer review','metric':'Accepted complete answers','baseline_pct':percentages['baseline'],
                     'multi_unlora_pct':percentages['contrast_8'],'change_pp':f'{delta:.1f}'}],
            'positive_only_pct':percentages['positive_8'],'accepted_counts':{name:int(value) for name,value in counts.items()},
            'paired_bootstrap_95_ci_pp':None if pair['accepted_paired_bootstrap_95_ci'] is None else [str(Decimal(v)*100) for v in pair['accepted_paired_bootstrap_95_ci']],
            'production_quality_verified':False}
    return data


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


def _confirmation_table(data):
    result=data.get('generated_confirmation')
    if result is None:return ''
    interval=result['paired_bootstrap_95_ci_pp']
    interval_text='not estimable' if interval is None else f"{Decimal(interval[0]):+.1f} to {Decimal(interval[1]):+.1f} pp"
    gate='Passed the internal factuality gate; broader release checks remain open.' if result['gate_passed'] else 'Did not pass the accuracy gate; this candidate is not the serving default.'
    return ('<section class="answer-review" aria-labelledby="answer-review-title"><h3 id="answer-review-title">Held-out answer review</h3>'
        +_table(result['rows'],caption=f"{result['n']} held-out adapter-evaluation questions · AI reference-grounded review.",label='Experimental candidate held-out answer review',corrected_label='Candidate v2')
        +'<div class="benchmark-notes"><p>'+gate+'</p><p>'
        +f"Matched positive-only control: {result['positive_only_pct']}% ({result['accepted_counts']['positive_8']}/{result['n']}). "
        +f"Candidate − base paired 95% interval: {interval_text}. All assigned questions stay in the denominator, including unfinished outputs.</p>"
        +'<p>This small, category-restricted internal review is not an official TruthfulQA score. Candidate v2 uses a third factual-reference adapter. '
        +'<a href="./results/reference_support_confirmation.json">Review results and limitations</a></p></div></section>')


def render_table(data):
    current = data["current"]
    mmlu, truth = current["rows"][:2]
    count_scope = (f'{mmlu["n"]}-question diagnostic per task' if mmlu["n"] == truth["n"]
                   else f'diagnostic · {mmlu["n"]} MMLU and {truth["n"]} TruthfulQA MC1 questions')
    invalid = truth["corrected_invalid_choices"]
    claim = ("See the source report for the validated scope." if current["production_quality_verified"] is True
             else "No accuracy or safety improvement is verified.")
    trial_links = [f'<a href="./results/{escape(report["source"])}">{escape(report["label"])}</a>' for report in data.get("trial_reports",[])]
    trial_link = '<details class="evaluation-details"><summary>Evaluation reports</summary><p>' + ' · '.join(trial_links) + '.</p></details>' if trial_links else ''
    return ('<section class="benchmark-panel panel" aria-labelledby="benchmark-title">'
        '<div class="panel-heading"><h2 id="benchmark-title">Benchmark results</h2>'
        '<span class="micro-label">MEASURED PILOT</span></div>'
        + _table(current["rows"], caption=f'Initial {count_scope} · {current["evaluation_protocol"]}.', label='Current measured pilot benchmark results')
        + '<div class="benchmark-notes"><p id="benchmark-model-note" hidden>The MMLU and TruthfulQA MC1 rows report the original pilot adapters.</p><p><strong>' + claim + '</strong> '
        'These are forced-choice label diagnostics, not general answer accuracy or a safety score. '
        f'TruthfulQA MC1 had {invalid} invalid corrected choice{"s" if invalid != 1 else ""}. '
        'AdvBench has not been measured; dashes mean unavailable.</p>'
        '<p>Change is measured in percentage points. '
        '<a href="./results/initial_pilot.json">Measured source JSON</a></p>' + trial_link + '</div>'
        +_confirmation_table(data)
        +'<details class="benchmark-history"><summary>Historical slide · unverified</summary>'
        '<p class="benchmark-history-scope">Original checkpoints and CSVs are unavailable. '
        'These supplied figures use different or unverified protocols and are not comparable to the current run.</p>'
        + _table(data["historical"]["rows"], caption='User-supplied historical reference · not reproduced.', label='Unverified historical slide figures')
        + '<p class="benchmark-history-source"><a href="./results/historical_slide.json">Historical source and limitations</a></p>'
        '</details></section>')


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
    return {"measured_rows":len(data["current"]["rows"]), "historical_rows":len(data["historical"]["rows"]),
            "changed_files":len(stale)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check",action="store_true")
    args = parser.parse_args()
    print(json.dumps(sync(check=args.check)))
