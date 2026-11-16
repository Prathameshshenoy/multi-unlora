from decimal import Decimal
import json
import shutil

import pytest
from scripts.sync_benchmarks import ROOT, build_data, sync


def test_user_site_has_one_bottom_table_with_scope_notes():
    assert sync(check=True)['changed_files'] == 0
    page = (ROOT/'web/index.html').read_text()
    section = page.split('<!-- BEGIN BENCHMARK RESULTS -->')[1].split('<!-- END BENCHMARK RESULTS -->')[0]
    assert section.count('<table ') == 1 and '<details' not in section
    assert page.index('class="method"') < page.index('id="benchmark-title"')
    assert 'latest live routing changes have not been evaluated' in section
    assert 'Display-guard refusals are excluded' in section
    assert 'Additional measurements' not in page and 'IMPLEMENTATION_PLAN' not in page


def test_table_values_match_compact_summary():
    page = (ROOT/'web/index.html').read_text()
    for row in build_data()['rows']:
        assert row['base_pct']+'%' in page and row['unlora_pct']+'%' in page
        assert f'n={row["n"]}' in page
    assert '−0.5 pp' in page


def test_summary_changes_require_resync_without_private_results(tmp_path):
    (tmp_path/'web/results').mkdir(parents=True)
    summary = tmp_path/'web/results/benchmarks.json'
    shutil.copyfile(ROOT/'web/results/benchmarks.json', summary)
    shutil.copyfile(ROOT/'web/index.html', tmp_path/'web/index.html')
    data = json.loads(summary.read_text())
    data['rows'][0]['base_pct'] = '60.0'
    summary.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='stale'):
        sync(tmp_path, check=True)
    sync(tmp_path)
    assert '−1.5 pp' in (tmp_path/'web/index.html').read_text()
    assert sync(tmp_path, check=True)['changed_files'] == 0


@pytest.mark.parametrize('value', ['NaN', '-1', '101'])
def test_invalid_percentage_cannot_be_published(tmp_path, value):
    (tmp_path/'web/results').mkdir(parents=True)
    data = build_data()
    data['rows'][0]['base_pct'] = value
    (tmp_path/'web/results/benchmarks.json').write_text(json.dumps(data))
    with pytest.raises(ValueError, match='Percentages'):
        build_data(tmp_path)


def test_readme_and_site_share_the_same_public_summary(tmp_path):
    (tmp_path/'web/results').mkdir(parents=True)
    shutil.copyfile(ROOT/'web/results/benchmarks.json', tmp_path/'web/results/benchmarks.json')
    shutil.copyfile(ROOT/'web/index.html', tmp_path/'web/index.html')
    readme = tmp_path/'README.md'
    readme.write_text('# Project\n\n<!-- BEGIN BENCHMARK RESULTS -->\nstale\n<!-- END BENCHMARK RESULTS -->\n')
    with pytest.raises(ValueError, match='README.*stale'):
        sync(tmp_path, check=True)
    sync(tmp_path)
    for row in build_data(tmp_path)['rows']:
        assert row['base_pct']+'%' in readme.read_text()
        assert row['unlora_pct']+'%' in readme.read_text()
        assert f"n={row['n']:,}" in readme.read_text()
    assert sync(tmp_path, check=True)['changed_files'] == 0
