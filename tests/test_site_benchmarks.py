from decimal import Decimal
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil

import pytest

from scripts.sync_benchmarks import ROOT, build_data, sync


class BenchmarkHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables, self.current, self.row, self.cell = [], None, None, None
        self.history_is_collapsed = None

    def handle_starttag(self, tag, attributes):
        attributes = dict(attributes)
        if tag == 'details' and attributes.get('class') == 'benchmark-history':
            self.history_is_collapsed = 'open' not in attributes
        if tag == 'table' and attributes.get('class') == 'benchmark-table':
            self.current = []
            self.tables.append(self.current)
        elif tag == 'tr' and self.current is not None:
            self.row = []
            self.current.append(self.row)
        elif tag in ('th','td') and self.row is not None:
            self.cell = ''

    def handle_data(self, text):
        if self.cell is not None:
            self.cell += text

    def handle_endtag(self, tag):
        if tag in ('th','td') and self.cell is not None:
            self.row.append(self.cell)
            self.cell = None
        elif tag == 'tr':
            self.row = None
        elif tag == 'table':
            self.current = None


def test_site_numbers_are_derived_from_measured_source_and_keep_missing_asr_missing():
    source = json.loads((ROOT/'docs/results/initial_pilot.json').read_text(),parse_float=Decimal)
    data = build_data()
    for row, key in zip(data['current']['rows'],('mmlu','truthfulqa_mc1')):
        summary = source['benchmarks'][key]['summary']
        assert Decimal(row['baseline_pct']) == summary['baseline_accuracy']*100
        assert Decimal(row['multi_unlora_pct']) == summary['unlora_accuracy']*100
        assert Decimal(row['change_pp']) == summary['paired_delta']*100
        assert row['n']==summary['n']==200
    assert data['current']['rows'][1]['corrected_invalid_choices']==1
    assert data['current']['rows'][2]['status']=='not_measured'
    assert all(data['current']['rows'][2][key] is None for key in ('baseline_pct','multi_unlora_pct','change_pp'))
    assert data['current']['forced_choice_gate_passed'] is False
    assert data['current']['production_quality_verified'] is False


def test_rendered_table_exposes_current_scope_and_separate_collapsed_history():
    assert sync(check=True)['changed_files']==0
    text = (ROOT/'web/index.html').read_text()
    document = BenchmarkHTML()
    document.feed(text)
    current, historical = document.tables[0],document.tables[-1]
    assert current[0]==['Benchmark','Metric','Base','Multi-UnLoRA','Change']
    assert current[1]==['MMLU','Label accuracy','59.0%','58.5%','−0.5 pp']
    assert current[2]==['TruthfulQA MC1','Label accuracy','58.0%','58.0%','0 pp']
    assert current[3]==['AdvBench','Attack success rate','—','—','—']
    assert document.history_is_collapsed
    assert [row[-1] for row in historical[1:]]==['−0.6 pp','−23.1 pp','+14.8 pp']
    assert 'not comparable to the current run' in text
    assert 'No accuracy or safety improvement is verified.' in text
    assert 'Initial 200-question diagnostic per task' in text
    assert './results/paired_trial.json' in text
    trial = json.loads((ROOT/'docs/results/paired_trial.json').read_text())
    assert build_data()['latest_trial']['status']==trial['status']


def test_source_changes_require_resync_and_update_the_public_numbers(tmp_path):
    for name in ('initial_pilot.json','historical_slide.json'):
        target = tmp_path/'docs/results'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/'docs/results'/name,target)
    (tmp_path/'web').mkdir()
    shutil.copyfile(ROOT/'web/index.html',tmp_path/'web/index.html')
    sync(tmp_path)
    source = tmp_path/'docs/results/initial_pilot.json'
    report = json.loads(source.read_text())
    report['benchmarks']['mmlu']['summary'].update(baseline_accuracy=.6,paired_delta=-.015)
    source.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='stale'):
        sync(tmp_path,check=True)
    sync(tmp_path)
    data = json.loads((tmp_path/'web/results/benchmarks.json').read_text())
    assert data['current']['rows'][0]['baseline_pct']=='60.0'
    assert data['current']['rows'][0]['change_pp']=='-1.5'
    assert sync(tmp_path,check=True)['changed_files']==0


def test_heldout_answer_table_uses_confirmation_not_consumed_development_score():
    report=json.loads((ROOT/'docs/results/reference_support_confirmation.json').read_text(),parse_float=Decimal)
    data=build_data()['generated_confirmation'];arms=report['baseline_comparison']['arms']
    assert data['n']==20
    assert Decimal(data['rows'][0]['baseline_pct'])==arms['baseline']['accepted_rate']*100
    assert Decimal(data['rows'][0]['multi_unlora_pct'])==arms['contrast_8']['accepted_rate']*100
    assert data['accepted_counts']=={'baseline':14,'contrast_8':14,'positive_8':16}
    assert data['gate_passed'] is False and data['production_quality_verified'] is False
    text=(ROOT/'web/index.html').read_text()
    assert 'not an official TruthfulQA score' in text
    assert 'Candidate v2' in text and 'not the serving default' in text
