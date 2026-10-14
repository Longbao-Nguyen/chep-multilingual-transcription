import argparse
import copy
import json
from pathlib import Path

import pytest

from scripts.kaggle_optimization.core import (boundary_issues, chunk_aligned_ranges, compare_segments, discover, edit_distance, errors, gpu_valid, invalid_reasons,
                                             merge, ranges, save, sha, validate_model)
from scripts.kaggle_optimization.cli import append_trial, cleanup, execute, notebook, prepare, read, report


def test_artifacts_and_nested_mount(tmp_path):
    path = tmp_path / 'datasets' / 'longnj' / 'french-whisper-ct2'
    path.mkdir(parents=True)
    (path / 'model.bin').write_bytes(b'model')
    files = {'model.bin': {'size': 5, 'sha256': sha(path / 'model.bin')}}
    assert discover(tmp_path, files) == path
    validate_model(path, files)
    (path / 'model.bin').write_bytes(b'wrong')
    with pytest.raises(ValueError):
        validate_model(path, files)
    with pytest.raises(ValueError):
        discover(tmp_path, files)


def test_ambiguous_mount(tmp_path):
    for name in ('one', 'two'):
        (tmp_path / name).mkdir()
        (tmp_path / name / 'model.bin').write_bytes(b'a')
    with pytest.raises(ValueError):
        discover(tmp_path, {'model.bin': {'size': 1, 'sha256': sha(tmp_path/'one/model.bin')}})


@pytest.mark.parametrize('overlap', [0, 5, 10])
def test_ranges(overlap):
    a, b = ranges(3600, overlap)
    assert a[0] == 0 and b[1] == 3600
    assert a[1] - b[0] == overlap
    assert a[1] - a[0] == b[1] - b[0]


def test_chunk_aligned_partition_preserves_global_chunk_grid():
    import math
    duration = 5955.5626875
    spans = chunk_aligned_ranges(duration, 40)
    assert spans == [(0, 2960), (2960, duration)]
    actual = [(start + i*40, min(stop, start+(i+1)*40)) for start, stop in spans
              for i in range(math.ceil((stop-start)/40))]
    expected = [(i*40, min(duration, (i+1)*40)) for i in range(math.ceil(duration/40))]
    assert actual == expected
    with pytest.raises(ValueError):
        chunk_aligned_ranges(20, 40)


@pytest.mark.parametrize('duration', [0, -1, float('nan'), float('inf')])
def test_bad_ranges(duration):
    with pytest.raises(ValueError):
        ranges(duration, 0)


def test_merge_only_cross_shard_overlap():
    left = [{'start': 1, 'end': 2, 'text': 'Oui'}, {'start': 48, 'end': 50, 'text': 'Oui'}]
    right = [{'start': 3, 'end': 5, 'text': 'Oui'}, {'start': 20, 'end': 21, 'text': 'Oui'}]
    merged, removed = merge([left, right], [(0, 55), (45, 100)])
    assert [s['start'] for s in merged] == [1, 48, 65]
    assert len(removed) == 1
    # Ambiguous partial phrases must survive for manual boundary review.
    right[0]['text'] = 'Oui mais'
    assert len(merge([left, right], [(0, 55), (45, 100)])[0]) == 4


def test_invalid_timestamps_and_missing_shard():
    with pytest.raises(ValueError):
        merge([[{'start': -1, 'end': 2, 'text': 'x'}]], [(0, 20)])
    with pytest.raises(ValueError):
        merge([], [(0, 20)])


def valid_job():
    return {'ok': True, 'gpu': 'Tesla T4', 'profile': {'device': 'cuda', 'compute_type': 'float16',
            'concurrency': 2}, 'fingerprint': 'fixed', 'versions': {'ct2': '4.8.2'},
            'files': {'txt': 'hash'}, 'inference_windows': [[10, 20]]}


def test_invalid_run_classification():
    a, b = valid_job(), valid_job()
    assert invalid_reasons([a, b], 'fixed', True) == []
    b['inference_windows'] = [[20, 30]]
    assert 'no observed inference overlap' in invalid_reasons([a, b], 'fixed', True)
    b['profile']['device'] = 'cpu'
    assert not gpu_valid('Tesla T4', b['profile'])
    assert invalid_reasons([b], 'fixed')
    assert invalid_reasons([a], 'wrong')
    assert invalid_reasons([], 'fixed')
    a['files'] = {}
    assert invalid_reasons([a], 'fixed')


def test_quality_reference():
    assert errors('École, oui!', 'école oui') == {'wer_percent': 0, 'cer_percent': 0}
    assert errors('un deux', 'un')['wer_percent'] == 50
    assert errors('', 'un')['wer_percent'] is None


def test_exact_edit_distance_against_small_dynamic_program():
    import random
    rng = random.Random(17)
    for _ in range(100):
        a = [rng.choice('abcd') for _ in range(rng.randrange(30))]
        b = [rng.choice('abcd') for _ in range(rng.randrange(30))]
        row = list(range(len(b) + 1))
        for i, token in enumerate(a, 1):
            current = [i]
            for j, other in enumerate(b, 1):
                current.append(min(current[-1] + 1, row[j] + 1, row[j-1] + (token != other)))
            row = current
        assert edit_distance(a, b) == row[-1]
    assert edit_distance('bonjour ' * 10000, 'bonjour ' * 10000 + 'oui') == 3


def test_timestamp_disagreement_and_omissions():
    baseline = [{'start': 1, 'end': 2, 'text': 'Bonjour'}, {'start': 3, 'end': 4, 'text': 'Au revoir'}]
    candidate = [{'start': 1.5, 'end': 2.25, 'text': 'Bonjour'}]
    result = compare_segments(baseline, candidate)
    assert not result['segments_exact']
    assert result['unmatched_baseline_segments'] == 1
    assert result['start_difference_seconds_max'] == .5
    assert result['end_difference_seconds_max'] == .25


def test_boundary_audit_flags_partial_repeat_without_changing_outputs():
    segments = [{'start': 40, 'end': 52, 'text': 'Voici six mois de lecture machine.'},
                {'start': 48, 'end': 60, 'text': 'Six mois de lecture machine et des frais.'}]
    before = copy.deepcopy(segments)
    issues = boundary_issues(segments, 50)
    assert issues[0]['words'] == 5
    assert segments == before
    segments[1]['start'] = 53
    assert boundary_issues(segments, 50) == []
    assert boundary_issues(before, 150) == []


def make_plan(tmp_path, duration=3600):
    audio = tmp_path / 'audio.wav'
    audio.write_bytes(b'fixture')
    out = tmp_path / 'experiment'
    prepare(argparse.Namespace(hypothesis='H1', model_dataset='longnj/french-whisper-ct2/1',
            duration=duration, overlap=5, output=out, audio=audio, owner='tester', repeats=3))
    return out, read(out/'plan.json')


@pytest.mark.parametrize('duration', [3, 3600])
def test_freeze_notebook_and_pending_report(tmp_path, duration):
    out, plan = make_plan(tmp_path, duration)
    assert len(plan['trials']) == 6
    assert [t['arm'] for t in plan['trials']] == ['A', 'B', 'B', 'A', 'A', 'B']
    assert plan['freeze']['config']['model_config']['sweet_spot']['concurrency'] == 2
    nb = notebook(plan, 'hf', (0, duration))
    compile(''.join(nb['cells'][0]['source']), '<notebook>', 'exec')
    report(out)
    assert read(out/'summary.json')['status'] == 'PENDING VALID BENCHMARK'


def test_cleanup_refuses_other_resources(tmp_path):
    save(tmp_path/'ownership.json', {'owner': 'tester', 'experiment': 'abc',
        'resources': [{'kind': 'datasets', 'ref': 'tester/production'}]})
    class FakeCLI:
        def identity(self):
            return 'tester'
        def run(self, *args, **kwargs):
            pytest.fail('Deletion must not be called')
    with pytest.raises(ValueError):
        cleanup(tmp_path, FakeCLI())


def test_report_does_not_mix_environments(tmp_path):
    out, plan = make_plan(tmp_path)
    for trial in plan['trials']:
        folder = out/trial['id']
        folder.mkdir()
        save(folder/'trial.json', {**trial, 'valid': True, 'fingerprint': plan['fingerprint'],
            'e2e': 100 if trial['arm'] == 'A' else 50, 'versions': {'ct2': trial['arm']}, 'gpu': 'T4'})
    report(out)
    summary = read(out/'summary.json')
    assert summary['status'] == 'PENDING VALID BENCHMARK'
    assert 'speedup' not in summary


def test_h2_controller_lifecycle_with_fake_remote(tmp_path, monkeypatch):
    """Exercise submission, overlap proof, retrieval, exports, ledger cleanup and report."""
    out, plan = make_plan(tmp_path)
    plan['hypothesis'] = 'H2'
    save(out/'plan.json', plan)
    submitted, polls, deleted = [], {}, []
    class FakeCLI:
        def __init__(self, root):
            pass
        def identity(self):
            return 'tester'
        def run(self, args, **kwargs):
            kind, action = args[:2]
            if action == 'delete':
                deleted.append(args[2])
            elif kind == 'datasets' and action == 'status':
                return 'ready', 0
            elif kind == 'kernels' and action == 'push':
                kernel = Path(args[args.index('-p')+1])
                ref = read(kernel/'kernel-metadata.json')['id']
                submitted.append(ref)
                polls[ref] = 0
            elif kind == 'kernels' and action == 'status':
                ref = args[2]
                # For dual trials, both shards must be pushed before the first status poll.
                trial_id = '-'.join(ref.split('-')[-3:-1])
                if trial_id.endswith('b'):
                    assert any(r == ref[:-1]+'1' for r in submitted)
                polls[ref] += 1
                status = 'RUNNING' if polls[ref] == 1 else 'COMPLETE'
                return f'has status "{status}"', 0
            elif kind == 'kernels' and action == 'output':
                path = Path(args[args.index('-p')+1])
                result = valid_job()
                result.update(fingerprint=plan['fingerprint'], python='3.12', nvidia_smi='fixture',
                              segments=[{'start': 1, 'end': 2, 'text': 'Bonjour'}],
                              timings={'model_source_prepare': 1})
                result['files'] = {}
                for ext in ('txt', 'srt'):
                    f = path/('transcript.'+ext)
                    f.write_text('fixture transcript', encoding='utf-8')
                    result['files'][ext] = {'sha256': sha(f), 'size': f.stat().st_size}
                save(path/'result.json', result)
            return '', 0
    monkeypatch.setattr('local_app.kaggle_cli.KaggleCLI', FakeCLI)
    monkeypatch.setattr('scripts.kaggle_optimization.cli.time.sleep', lambda _: None)
    execute(argparse.Namespace(directory=out, cli_root=tmp_path, gpu_eligibility_confirmed=True, reference=None))
    assert len(submitted) == 9
    assert len(deleted) == 15  # 9 kernels and 6 private audio datasets
    assert all(r['deleted'] for r in read(out/'ownership.json')['resources'])
    summary = read(out/'summary.json')
    assert summary['status'] == 'INCONCLUSIVE'
    assert all(summary['arms'][a]['valid_trials'] == 3 for a in ('A', 'B'))
    merged = read(out/'01-B/segments.json')
    assert merged[1]['start'] == 1798.5
    execute(argparse.Namespace(directory=out, cli_root=tmp_path, gpu_eligibility_confirmed=True, reference=None, resume=True))
    assert len(submitted) == 9  # Completed work must not run again.
    original_fingerprint = read(out/'plan.json')['fingerprint']
    append_trial(out, 'B')
    assert read(out/'plan.json')['fingerprint'] == original_fingerprint
    assert read(out/'plan.json')['trials'][-1]['id'] == '04-B'
