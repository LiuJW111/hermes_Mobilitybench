"""A model continuation must preserve original attempts and case provenance."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from run_hermes_batch import save_selected, is_provider_error


def test_continuation_keeps_old_attempts_and_source_models(tmp_path):
    old = [{'case_id': 'case_1', 'model_name': 'old', 'error': None},
           {'case_id': 'case_2', 'model_name': 'old', 'error': 'quota'},
           {'case_id': 'case_3', 'model_name': 'old', 'error': 'task incomplete'}]
    save_selected(tmp_path, old, old, 'old')
    old_file = tmp_path/'results/old/raw_results.json'
    original_bytes = old_file.read_bytes()
    new = [{'case_id': 'case_2', 'model_name': 'new', 'error': None}]
    chosen = [old[0], new[0], old[2]]
    save_selected(tmp_path, chosen, new, 'new')
    assert old_file.read_bytes() == original_bytes
    assert json.loads((tmp_path/'selected_results.json').read_text(encoding='utf-8')) == chosen
    assert json.loads((tmp_path/'results/new/raw_results.json').read_text(encoding='utf-8')) == new


def test_bus_numbers_are_not_provider_status_codes():
    assert not is_provider_error('无法查询华章名苑，请补充地址以避开401路公交。')
    assert not is_provider_error('请换乘403路，坐标116.429,39.401')
    assert is_provider_error("Hermes conversation failed: Error code: 403 - insufficient_quota")
    assert is_provider_error('HTTP 429 Too Many Requests')
