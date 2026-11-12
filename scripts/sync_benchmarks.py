"""Render one website table from the compact public evaluation summary.

Raw outputs, private measurements and research reports are never read or copied.
"""
import argparse
from decimal import Decimal
from html import escape
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BEGIN = '<!-- BEGIN BENCHMARK RESULTS -->'
END = '<!-- END BENCHMARK RESULTS -->'


def build_data(root=ROOT):
    data = json.loads((Path(root)/'web/results/benchmarks.json').read_text())
    if set(data) != {'protocol', 'rows', 'notes'}:
        raise ValueError('Only aggregate table rows and scope notes may be published')
    if data.get('protocol') != 'public-evaluation-summary-v1' or not data.get('rows'):
        raise ValueError('A public evaluation summary is required')
    for row in data['rows']:
        if set(row) != {'benchmark','metric','n','base_pct','unlora_pct'}:
            raise ValueError('Only aggregate table values may be published')
        if type(row['n']) is not int or row['n'] <= 0:
            raise ValueError('Each result needs an explicit sample size')
        for key in ('base_pct', 'unlora_pct'):
            value = row[key]
            if not isinstance(value, str) or not Decimal(value).is_finite() or not 0 <= Decimal(value) <= 100:
                raise ValueError('Percentages must be finite values from 0 to 100')
    if not data.get('notes') or any(not isinstance(n, str) or not n.strip() for n in data['notes']):
        raise ValueError('Evaluation scope notes are required')
    return data


def render_table(data):
    body = []
    for row in data['rows']:
        delta = Decimal(row['unlora_pct']) - Decimal(row['base_pct'])
        change = '0 pp' if delta == 0 else f"{'+' if delta > 0 else '−'}{abs(delta):.1f} pp"
        body.append('<tr><th scope="row">' + escape(row['benchmark']) + f' (n={row["n"]})</th>'
            '<td class="benchmark-metric">' + escape(row['metric']) + '</td>'
            '<td>' + escape(row['base_pct']) + '%</td>'
            '<td>' + escape(row['unlora_pct']) + '%</td><td>' + change + '</td></tr>')
    return ('<section class="benchmark-panel panel" aria-labelledby="benchmark-title">'
        '<div class="panel-heading"><h2 id="benchmark-title">Evaluation results</h2></div>'
        '<div class="benchmark-scroll" role="region" tabindex="0" aria-label="Model evaluation results">'
        '<table class="benchmark-table"><caption>Recorded evaluation profiles</caption>'
        '<thead><tr><th scope="col">Benchmark</th><th scope="col" class="benchmark-metric">Metric</th>'
        '<th scope="col">Base</th><th scope="col">Multi-UnLoRA</th><th scope="col">Change</th></tr></thead>'
        '<tbody>' + ''.join(body) + '</tbody></table></div><div class="benchmark-notes">'
        + ''.join('<p>' + escape(note) + '</p>' for note in data['notes']) + '</div></section>')


def sync(root=ROOT, *, check=False):
    root = Path(root)
    index = root/'web/index.html'
    text = index.read_text()
    if text.count(BEGIN) != 1 or text.count(END) != 1:
        raise ValueError('The index needs one benchmark-generation marker pair')
    data = build_data(root)
    before, remainder = text.split(BEGIN, 1)
    _, after = remainder.split(END, 1)
    rendered = before + BEGIN + '\n    ' + render_table(data) + '\n    ' + END + after
    changed = text != rendered
    if check and changed:
        raise ValueError('Site benchmark results are stale; run scripts/sync_benchmarks.py')
    if changed and not check:
        index.write_text(rendered)
    return {'measured_rows':len(data['rows']), 'changed_files':int(changed)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    print(json.dumps(sync(check=parser.parse_args().check)))
