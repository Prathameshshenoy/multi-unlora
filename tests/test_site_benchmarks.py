from decimal import Decimal
import json
import shutil

import pytest
from scripts.sync_benchmarks import ROOT, build_data, sync


def test_tables_match_source_scores_and_keep_unknown_full_asr_missing():
    source = json.loads((ROOT/'docs/results/initial_pilot.json').read_text(), parse_float=Decimal)
    data = build_data()
    for row, key in zip(data['current']['rows'], ('mmlu', 'truthfulqa_mc1')):
        score = source['benchmarks'][key]['summary']
        assert Decimal(row['baseline_pct']) == score['baseline_accuracy']*100
        assert Decimal(row['multi_unlora_pct']) == score['unlora_accuracy']*100
        assert row['n'] == score['n']
    assert data['current']['rows'][2]['baseline_pct'] is None
    assert data['current']['rows'][2]['status'] == 'not_measured'


def test_user_facing_site_omits_old_figures_and_internal_plans():
    assert sync(check=True)['changed_files'] == 0
    page = (ROOT/'web/index.html').read_text()
    assert 'Historical slide' not in page and 'unverified' not in page
    assert '64.6%' not in page and '50.6%' not in page
    assert 'IMPLEMENTATION_PLAN' not in page and 'Vercel' not in page
    assert 'Additional measurements' in page
    assert 'Website guard refusals were not counted' in page


def test_source_changes_require_resync(tmp_path):
    source = tmp_path/'docs/results/initial_pilot.json'; source.parent.mkdir(parents=True)
    shutil.copyfile(ROOT/'docs/results/initial_pilot.json', source)
    (tmp_path/'web').mkdir(); shutil.copyfile(ROOT/'web/index.html', tmp_path/'web/index.html')
    sync(tmp_path)
    report = json.loads(source.read_text())
    report['benchmarks']['mmlu']['summary'].update(baseline_accuracy=.6, paired_delta=-.015)
    source.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='stale'):
        sync(tmp_path, check=True)
    sync(tmp_path)
    assert build_data(tmp_path)['current']['rows'][0]['baseline_pct'] == '60.0'
    assert sync(tmp_path, check=True)['changed_files'] == 0


def test_factual_profile_is_separate_from_default_model():
    data = build_data()['selective_confirmation']
    assert data['accepted_counts'] == {'baseline':17,'selective_2':17,'positive_1':16,'positive_2':11}
    assert data['production_quality_verified'] is False
    page = (ROOT/'web/index.html').read_text()
    assert 'rather than the default demo settings' in page
    assert 'additional factual-reference adapter' in page


def test_raw_safety_subset_preserves_denominator_and_uncertainty():
    data = build_data()['raw_safety_development']
    assert data['n'] == 20
    assert data['baseline_pct'] == data['multi_unlora_pct'] == '0.0'
    assert Decimal(data['upper_95_pct']) > 10
    assert data['production_quality_verified'] is False
