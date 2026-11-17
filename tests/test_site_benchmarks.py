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
    for note in build_data()['notes']:
        assert note in section
    assert 'Display-guard refusals are excluded' in section
    assert 'Additional measurements' not in page and 'IMPLEMENTATION_PLAN' not in page


def test_table_values_match_compact_summary():
    page = (ROOT/'web/index.html').read_text()
    for row in build_data()['rows']:
        assert row['base_pct']+'%' in page and row['unlora_pct']+'%' in page
        assert f'n={row["n"]:,}' in page
    assert '<th scope="col">Change</th>' in page


def test_summary_changes_require_resync_without_private_results(tmp_path):
    (tmp_path/'web/results').mkdir(parents=True)
    summary = tmp_path/'web/results/benchmarks.json'
    shutil.copyfile(ROOT/'web/results/benchmarks.json', summary)
    shutil.copyfile(ROOT/'web/index.html', tmp_path/'web/index.html')
    data = json.loads(summary.read_text())
    data['rows'][0]['base_pct'] = '60.0'
    data['rows'][0].pop('base_count',None)
    data['rows'][0].pop('unlora_count',None)
    summary.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='stale'):
        sync(tmp_path, check=True)
    sync(tmp_path)
    assert '60.0%' in (tmp_path/'web/index.html').read_text()
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


@pytest.mark.parametrize('baseline,corrected,html_change,markdown_change', [
    ('30.0','7.5','−22.5 pp','-22.5 pp'),
    ('41.2','55.6','+14.4 pp','+14.4 pp'),
    ('60.0','60.0','0 pp','0 pp'),
])
def test_tables_report_percentage_point_changes(baseline, corrected, html_change, markdown_change):
    from scripts.sync_benchmarks import render_table, render_markdown
    data = {'rows':[{'benchmark':'Example', 'metric':'Accuracy', 'n':100,
                     'base_pct':baseline, 'unlora_pct':corrected}], 'notes':['Test fixture.']}
    assert html_change in render_table(data)
    assert markdown_change in render_markdown(data)


def test_changes_use_exact_counts_instead_of_rounded_endpoints():
    from scripts.sync_benchmarks import render_table, render_markdown
    data={'rows':[{'benchmark':'Example','metric':'Accuracy','n':6,'base_count':4,
                   'unlora_count':5,'base_pct':'66.7','unlora_pct':'83.3'}],'notes':['Test fixture.']}
    assert '+16.7 pp' in render_table(data)
    assert '+16.7 pp' in render_markdown(data)
