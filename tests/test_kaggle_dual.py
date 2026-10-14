import json
import hashlib
import wave

import pytest

from local_app.catalog import selection
from local_app.config import prepare_data
from local_app.kaggle_backend import KaggleBackend
from local_app.kaggle_merge import merge_segments, shard_ranges
from local_app.store import Store
from local_app.worker import select_range


def test_21_minute_boundary_and_five_second_overlap():
    with pytest.raises(ValueError):
        shard_ranges(21 * 60 - .001)
    spans = shard_ranges(21 * 60)
    assert spans[0][0] == 0
    assert spans[1][1] == 21 * 60
    assert spans[0][1] - spans[1][0] == 5


def test_merge_removes_only_certain_repeated_prefix_at_seam():
    spans = shard_ranges(22 * 60)
    left = [{'start': 648, 'end': 662.5,
             'text': 'Nous avons six mois de lecture machine, et 573 000 euros.'}]
    right = [{'start': 0, 'end': 20,
              'text': 'Six mois de lecture machine et 573 000 euros de réactif seul.'}]
    merged, audit = merge_segments([left, right], spans)
    assert merged[0]['text'] == left[0]['text']
    assert merged[1]['text'] == 'de réactif seul.'
    assert audit == [{'left_start': 648, 'right_start': spans[1][0], 'removed_words': 9,
                      'lexical_confidence': 1.0, 'decision': 'remove_exact_overlapping_right_prefix'}]


def test_merge_keeps_short_ambiguous_and_nonoverlapping_repetitions():
    spans = shard_ranges(22 * 60)
    left = [{'start': 650, 'end': 662, 'text': 'Oui, oui, oui.'}]
    right = [{'start': 0, 'end': 2, 'text': 'Oui oui oui. Ensuite.'}]
    merged, audit = merge_segments([left, right], spans)
    assert len(merged) == 2 and not audit
    left = [{'start': 648, 'end': 662.5, 'text': 'six mois de lecture machine et 573 000 euros'}]
    right = [{'start': 0, 'end': 20, 'text': 'six mois de lecture humaine et 573 000 euros de réactif'}]
    merged, audit = merge_segments([left, right], spans)
    assert merged[1]['text'] == right[0]['text'] and not audit
    # Identical words at different times may be real repetition.
    left = [{'start': 640, 'end': 652, 'text': 'un deux trois quatre cinq'}]
    right = [{'start': 0, 'end': 5, 'text': 'un deux trois quatre cinq et six'}]
    merged, audit = merge_segments([left, right], spans)
    assert merged[1]['text'] == right[0]['text'] and not audit


def test_merge_rejects_malformed_remote_segments():
    spans = shard_ranges(22 * 60)
    for malformed in ({'start': 0, 'end': 1, 'text': 'hello'},
                      [{'start': True, 'end': 1, 'text': 'hello'}],
                      [{'start': 0, 'end': 1}]):
        with pytest.raises(ValueError, match='Invalid Kaggle shard'):
            merge_segments([[], malformed], spans)


def test_select_range_preserves_exact_pcm_samples(tmp_path):
    audio = tmp_path / 'audio.wav'
    data = bytes(range(256)) * 1250
    with wave.open(str(audio), 'wb') as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(data)
    assert select_range(audio, 2.5, 7.5, 10) == 5
    with wave.open(str(audio), 'rb') as selected:
        assert selected.getnframes() == 5 * 16000
        assert selected.readframes(5 * 16000) == data[2 * 16000 * 2 + 16000:7 * 16000 * 2 + 16000]


@pytest.mark.parametrize('duration,dual', [(21 * 60 - 1, False), (21 * 60, True)])
def test_foreign_account_attaches_public_model_and_only_owns_its_inputs(tmp_path, monkeypatch, duration, dual):
    class Submitted(Exception):
        pass

    store = Store(prepare_data(tmp_path))
    store.add('job', 'sample.wav', 5, selection('en', 'kaggle'))
    (tmp_path / 'sources/job').write_bytes(b'audio')
    backend = KaggleBackend(store)
    monkeypatch.setattr('local_app.worker.decoded_duration', lambda _: duration)
    monkeypatch.setattr(backend.cli, 'identity', lambda: 'otheruser')
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        if args[:2] == ['datasets', 'status']:
            return ('ready', 0) if any(call[:2] == ['datasets', 'create'] for call in calls) else ('missing', 1)
        if args[:2] == ['kernels', 'status']:
            return 'missing', 1
        if args[:2] == ['kernels', 'push']:
            raise Submitted()
        return '', 0

    monkeypatch.setattr(backend.cli, 'run', run)
    with pytest.raises(Submitted):
        backend.run(store.get('job'), lambda: False)
    remote = json.loads(store.get('job')['remote'])
    folder = tmp_path / 'work/job/kernel' / ('0' if dual else '')
    metadata = json.loads((folder / 'kernel-metadata.json').read_text(encoding='utf-8'))
    assert metadata['id'].startswith('otheruser/chep-run-')
    assert metadata['dataset_sources'] == [remote['dataset'], 'longnj/english-whisper-ct2/1']
    assert remote['dataset'].startswith('otheruser/chep-input-')
    assert remote['dual'] is dual
    backend.cleanup('job')
    deleted = [args[2] for args in calls if args[1] == 'delete']
    assert deleted and all(ref.startswith('otheruser/') for ref in deleted)
    assert json.loads(store.get('job')['remote'])['cleaned']


def test_dual_backend_merges_verified_shards_and_cleans_both_kernels(tmp_path, monkeypatch):
    store = Store(prepare_data(tmp_path))
    config = selection('en', 'kaggle')
    store.add('job', 'sample.wav', 5, config)
    (tmp_path / 'sources/job').write_bytes(b'audio')
    backend = KaggleBackend(store)
    monkeypatch.setattr('local_app.worker.decoded_duration', lambda _: 22 * 60)
    monkeypatch.setattr(backend.cli, 'identity', lambda: 'otheruser')
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if args[:2] == ['datasets', 'status']:
            return ('ready', 0) if any(call[:2] == ['datasets', 'create'] for call in calls) else ('missing', 1)
        if args[:2] == ['kernels', 'status']:
            return 'x has status "KernelWorkerStatus.COMPLETE"', 0
        if args[:2] == ['kernels', 'output']:
            folder = tmp_path / 'work/job/results' / args[2][-1]
            folder.mkdir(parents=True, exist_ok=True)
            index = int(args[2][-1])
            remote = json.loads(store.get('job')['remote'])
            stem = f'job-{index}'
            segments = ([{'start': 648, 'end': 662.5, 'text': 'six mois de lecture machine, et 573 000 euros'}]
                        if index == 0 else
                        [{'start': 0, 'end': 20,
                          'text': 'six mois de lecture machine et 573 000 euros de réactif seul'}])
            result = {'schema_version': 1, 'job_id': 'job', 'source_sha256': remote['source_sha256'],
                      'part_index': index, 'span': remote['spans'][index],
                      'source_duration': 22 * 60, 'model_dataset': 'longnj/english-whisper-ct2/1',
                      'ok': True, 'profile': {'device': 'cuda', 'provider': 'kaggle',
                      'device_name': 'Tesla T4', 'language': 'en', 'revision': config['model_config']['revision'],
                      **config['model_config']['sweet_spot']}, 'files': {}}
            for ext, content in (('txt', 'text'), ('srt', 'subtitle'),
                                 ('segments.json', json.dumps(segments))):
                data = content.encode('utf-8')
                (folder / f'{stem}.{ext}').write_bytes(data)
                result['files'][ext] = {'sha256': hashlib.sha256(data).hexdigest(), 'size': len(data)}
            (folder / 'result.json').write_text(json.dumps(result), encoding='utf-8')
        return '', 0

    monkeypatch.setattr(backend.cli, 'run', fake_run)
    backend.run(store.get('job'), lambda: False)
    assert store.get('job')['status'] == 'COMPLETED'
    text = (tmp_path / 'outputs/job.txt').read_text(encoding='utf-8')
    assert text.count('six mois de lecture machine') == 1
    assert 'de réactif seul' in text
    assert (tmp_path / 'outputs/job.srt').is_file()
    assert not (tmp_path / 'sources/job').exists()
    remote = json.loads(store.get('job')['remote'])
    assert remote['cleaned'] and len(remote['kernels']) == 2
    assert len([args for args in calls if args[:2] == ['kernels', 'delete']]) == 2
