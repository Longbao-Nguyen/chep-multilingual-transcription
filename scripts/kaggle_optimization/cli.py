"""python -m scripts.kaggle_optimization.cli --help"""
import argparse
import csv
import hashlib
import json
import re
import shutil
import statistics
import time
import uuid
from pathlib import Path
from .core import boundary_issues, chunk_aligned_ranges, compare_segments, errors, invalid_reasons, merge, ranges, save, sha

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def prepare(args):
    if args.hypothesis == 'H1' and not re.fullmatch(r'longnj/french-whisper-ct2/[1-9][0-9]*', args.model_dataset or ''):
        raise ValueError('H1 requires --model-dataset longnj/french-whisper-ct2/VERSION')
    strategy = getattr(args, 'split_strategy', 'midpoint')
    ranges(args.duration, args.overlap if args.hypothesis == 'H2' and strategy == 'midpoint' else 0)
    if strategy == 'chunk-aligned' and (args.hypothesis != 'H2' or args.overlap != 0):
        raise ValueError('Chunk-aligned control requires H2 and zero overlap')
    cfg = read(ROOT / 'config/transcription_models.json')['languages']['fr']
    if strategy == 'chunk-aligned':
        chunk_aligned_ranges(args.duration, cfg['chunk_seconds'])
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    files = {'local_app/' + n: (ROOT / 'local_app' / n).read_text(encoding='utf-8')
             for n in ('__init__.py', 'worker.py', 'config.py', 'hardware.py')}
    for n in ('core.py', 'runtime.py'):
        files[n] = Path(__file__).with_name(n).read_text(encoding='utf-8')
    # Identical inference pins in each arm; inventory comparison also gates transitive packages.
    pins = ('faster-whisper', 'ctranslate2', 'av', 'numpy', 'huggingface_hub', 'tokenizers',
            'onnxruntime', 'tqdm', 'PyYAML', 'filelock', 'fsspec', 'packaging', 'requests',
            'flatbuffers', 'protobuf', 'certifi', 'charset-normalizer', 'idna', 'urllib3')
    files['requirements.txt'] = '\n'.join(line for line in
        (ROOT / 'requirements-lock.txt').read_text().splitlines() if line.split('==')[0] in pins) + '\n'
    freeze = {'config': {'language': 'fr', 'model_config': cfg}, 'worker_files': files,
              'audio_sha256': sha(args.audio), 'duration': args.duration}
    plan = {'schema_version': 1, 'experiment': uuid.uuid4().hex[:16], 'owner': args.owner,
            'hypothesis': args.hypothesis, 'audio': str(args.audio.resolve()),
            'model_dataset': args.model_dataset, 'overlap': args.overlap,
            'split_strategy': strategy,
            'fingerprint': digest(freeze), 'freeze': freeze, 'trials': []}
    for repetition in range(args.repeats):
        arms = ['A', 'B'] if repetition % 2 == 0 else ['B', 'A']
        for arm in arms:
            plan['trials'].append({'id': f'{repetition+1:02}-{arm}', 'arm': arm,
                                   'repetition': repetition + 1})
    save(directory / 'plan.json', plan)
    save(directory / 'ownership.json', {'owner': args.owner, 'experiment': plan['experiment'], 'resources': []})
    print(directory / 'plan.json')


def notebook(plan, source, span):
    frozen = plan['freeze']
    spec = {k: v for k, v in frozen.items() if k != 'worker_files'}
    spec.update(fingerprint=plan['fingerprint'], source=source, span=span)
    files = {**frozen['worker_files'], 'spec.json': json.dumps(spec)}
    cell = "import os, sys, subprocess\nfrom pathlib import Path\n"
    cell += "os.environ['HF_HUB_DISABLE_XET']='1'\nos.environ['TOKENIZERS_PARALLELISM']='false'\n"
    cell += "root=Path('/tmp/chep-optimization')\nroot.mkdir(exist_ok=True)\n"
    cell += f'files={files!r}\n'
    cell += "for name, content in files.items():\n p=root/name\n p.parent.mkdir(parents=True,exist_ok=True)\n p.write_text(content,encoding='utf-8')\n"
    cell += "subprocess.run([sys.executable,str(root/'runtime.py')],cwd=root,check=True)\n"
    return {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {'kernelspec': {
        'display_name': 'Python 3', 'name': 'python3', 'language': 'python'}}, 'cells': [
        {'id': 'benchmark', 'cell_type': 'code', 'metadata': {}, 'execution_count': None,
         'outputs': [], 'source': cell.splitlines(True)}]}


def cleanup(directory, cli):
    from local_app.kaggle_backend import cleanup_response
    ledger = read(directory / 'ownership.json')
    if cli.identity() != ledger['owner']:
        raise ValueError('Cleanup account mismatch')
    pattern = re.escape(ledger['owner'] + '/chep-opt-' + ledger['experiment']) + r'-[0-9]{2}-[ab]-(?:input|[01])'
    for resource in reversed(ledger['resources']):
        if resource.get('deleted'):
            continue
        if resource['kind'] not in ('kernels', 'datasets') or not re.fullmatch(pattern, resource['ref']):
            raise ValueError('Refusing unowned resource')
        output, code = cli.run([resource['kind'], 'delete', resource['ref'], '-y'], check=False)
        if code and cleanup_response(output, resource['kind'], resource['ref']) != 'missing':
            raise ValueError('Cleanup failed; resource retained in ownership.json')
        resource['deleted'] = True
        save(directory / 'ownership.json', ledger)


def execute(args):
    from local_app.kaggle_cli import KaggleCLI
    from local_app.kaggle_backend import kernel_status
    from local_app.worker import stamp
    directory = args.directory.resolve()
    plan = read(directory / 'plan.json')
    cli = KaggleCLI(args.cli_root)
    if cli.identity() != plan['owner']:
        raise ValueError('Authenticated account differs from frozen owner')
    if not args.gpu_eligibility_confirmed:
        raise ValueError('PENDING: confirm account verification and usable GPU quota')
    ledger = read(directory / 'ownership.json')
    def own(kind, ref):
        ledger['resources'].append({'kind': kind, 'ref': ref, 'deleted': False})
        save(directory / 'ownership.json', ledger)
    for trial in plan['trials']:
        folder = directory / trial['id']
        if folder.exists():
            if getattr(args, 'resume', False):
                previous = read(folder / 'trial.json')
                prefix = plan['owner'] + '/chep-opt-' + plan['experiment'] + '-' + trial['id'].lower() + '-'
                owned = [r for r in ledger['resources'] if r['ref'].startswith(prefix)]
                if 'attempt_elapsed' in previous and all(r.get('deleted') for r in owned):
                    continue  # Keep failed evidence; never resubmit an attempted job.
            raise ValueError('Trial already attempted. Use a new experiment; never blindly resubmit.')
        folder.mkdir()
        record = {**trial, 'hypothesis': plan['hypothesis'], 'fingerprint': plan['fingerprint'],
                  'duration': plan['freeze']['duration'], 'valid': False, 'retries': 0,
                  'started_at': time.time(), 'jobs': [], 'events': []}
        save(folder / 'trial.json', record)
        begin = time.perf_counter()
        try:
            base = plan['owner'] + '/chep-opt-' + plan['experiment'] + '-' + trial['id'].lower()
            dataset = base + '-input'
            inputs = folder / 'input'
            inputs.mkdir()
            shutil.copyfile(plan['audio'], inputs / 'benchmark.audio')
            if sha(inputs / 'benchmark.audio') != plan['freeze']['audio_sha256']:
                raise ValueError('Audio changed after freezing')
            save(inputs / 'dataset-metadata.json', {'id': dataset, 'title': dataset.split('/')[1],
                                                   'licenses': [{'name': 'other'}]})
            own('datasets', dataset)
            cli.run(['datasets', 'create', '-p', str(inputs)], timeout=3600)
            deadline = time.monotonic() + 1800
            while True:
                status, code = cli.run(['datasets', 'status', dataset], check=False)
                if not code and status.strip().lower() == 'ready':
                    break
                if time.monotonic() > deadline or status.strip().lower() in ('error', 'failed'):
                    raise ValueError('Audio dataset unavailable')
                time.sleep(10)
            record['audio_prepare_upload_ready'] = time.perf_counter() - begin
            dual = plan['hypothesis'] == 'H2' and trial['arm'] == 'B'
            if dual and plan.get('split_strategy') == 'chunk-aligned':
                spans = chunk_aligned_ranges(record['duration'], plan['freeze']['config']['model_config']['chunk_seconds'])
            else:
                spans = ranges(record['duration'], plan['overlap']) if dual else [(0, record['duration'])]
            source = 'dataset' if plan['hypothesis'] == 'H1' and trial['arm'] == 'B' else 'hf'
            # Submit both before polling either. Actual overlap, not submission order, determines validity.
            for index, span in enumerate(spans):
                kernel = folder / str(index)
                kernel.mkdir()
                ref = base + '-' + str(index)
                sources = [dataset + '/1'] + ([plan['model_dataset']] if source == 'dataset' else [])
                save(kernel / 'run.ipynb', notebook(plan, source, span))
                save(kernel / 'kernel-metadata.json', {'id': ref, 'title': ref.split('/')[1],
                    'code_file': 'run.ipynb', 'language': 'python', 'kernel_type': 'notebook',
                    'is_private': True, 'enable_gpu': True, 'enable_internet': True,
                    'machine_shape': 'NvidiaTeslaT4', 'dataset_sources': sources,
                    'competition_sources': [], 'kernel_sources': [], 'model_sources': []})
                own('kernels', ref)
                record['jobs'].append({'ref': ref, 'submitted_at': time.time(), 'span': span})
                save(folder / 'trial.json', record)
                cli.run(['kernels', 'push', '-p', str(kernel), '--accelerator', 'NvidiaTeslaT4', '-t', '14400'], timeout=180)
            pending = set(range(len(spans)))
            deadline = time.monotonic() + 5 * 3600
            while pending:
                for index in sorted(pending):
                    job = record['jobs'][index]
                    status = kernel_status(cli.run(['kernels', 'status', job['ref']])[0])
                    now = time.time()
                    record['events'].append({'job': index, 'status': status, 'at': now})
                    if status == 'RUNNING':
                        job.setdefault('running_observed_at', now)
                    if status == 'COMPLETE':
                        job['complete_observed_at'] = now
                        pending.remove(index)
                    elif status is None or any(s in status for s in ('ERROR', 'FAILED', 'CANCEL')):
                        diagnostic = folder / str(index) / 'diagnostic'
                        diagnostic.mkdir(exist_ok=True)
                        cli.run(['kernels', 'output', job['ref'], '-p', str(diagnostic),
                                 '--file-pattern', r'^result\.json$', '--force'], check=False)
                        raise ValueError('Notebook failed or status unknown')
                save(folder / 'trial.json', record)
                if time.monotonic() > deadline:
                    raise TimeoutError('Notebook deadline exceeded')
                if pending:
                    time.sleep(10)
            retrieval = time.perf_counter()
            results = []
            for index, job in enumerate(record['jobs']):
                output = folder / str(index) / 'output'
                output.mkdir()
                cli.run(['kernels', 'output', job['ref'], '-p', str(output), '--file-pattern',
                         r'^(result\.json|transcript\.(txt|srt))$', '--force'], timeout=600)
                result = read(output / 'result.json')
                for ext in ('txt', 'srt'):
                    meta = result.get('files', {}).get(ext, {})
                    target = output / ('transcript.' + ext)
                    if not target.is_file() or target.stat().st_size != meta.get('size') or sha(target) != meta.get('sha256'):
                        raise ValueError('Output integrity failure')
                results.append(result)
            record['output_retrieval'] = time.perf_counter() - retrieval
            reasons = invalid_reasons(results, plan['fingerprint'], dual)
            if any('running_observed_at' not in j for j in record['jobs']):
                reasons.append('missing observed running interval')
            record['invalid_reasons'] = reasons
            merge_start = time.perf_counter()
            segments, removed = merge([r['segments'] for r in results], spans)
            save(folder / 'segments.json', segments)
            seam = spans[1][0] if dual and plan.get('split_strategy') == 'chunk-aligned' else record['duration']/2
            save(folder / 'boundary-audit.json', {'review': 'PENDING' if dual else 'not applicable', 'seam': seam,
                 'removed': removed, 'segments': [s for s in segments if
                     s['end'] >= seam-30 and s['start'] <= seam+30]})
            for ext in ('txt', 'srt'):
                if not dual:
                    shutil.copyfile(folder / '0' / 'output' / ('transcript.' + ext), folder / ('transcript.' + ext))
                else:
                    lines = [f"[{stamp(s['start'])} - {stamp(s['end'])}] {s['text']}\n" if ext == 'txt' else
                             f"{i}\n{stamp(s['start'], True)} --> {stamp(s['end'], True)}\n{s['text']}\n\n"
                             for i, s in enumerate(segments, 1)]
                    (folder / ('transcript.' + ext)).write_text(''.join(lines), encoding='utf-8')
            record['merge'] = time.perf_counter() - merge_start
            record['e2e'] = time.perf_counter() - begin
            record['rtf'] = record['e2e'] / record['duration']
            record['model_source_prepare'] = sum(r['timings']['model_source_prepare'] for r in results)
            record['gpu_session_minutes_observed'] = sum(j['complete_observed_at'] - j.get('running_observed_at', j['complete_observed_at']) for j in record['jobs']) / 60
            record['versions'] = results[0]['versions']
            record['gpu'] = results[0]['gpu']
            record['python'] = results[0]['python']
            record['nvidia_smi'] = results[0]['nvidia_smi']
            record['job_submit_to_running_observed'] = [
                j['running_observed_at'] - j['submitted_at'] for j in record['jobs'] if 'running_observed_at' in j]
            record['job_timings'] = [r['timings'] for r in results]
            record['quality'] = errors(args.reference.read_text(encoding='utf-8'), ' '.join(s['text'] for s in segments)) if args.reference else None
            record['valid'] = not reasons
        except (Exception, KeyboardInterrupt) as exc:
            record['invalid_reasons'] = [type(exc).__name__]
        finally:
            record['attempt_elapsed'] = time.perf_counter() - begin
            save(folder / 'trial.json', record)
        clean_start = time.perf_counter()
        try:
            cleanup(directory, cli)
        except Exception as exc:
            record['cleanup_error_type'] = type(exc).__name__
        ledger = read(directory / 'ownership.json')
        record['cleanup'] = time.perf_counter() - clean_start
        save(folder / 'trial.json', record)
        # Invalid evidence never becomes a speedup; stop before consuming more quota.
        if not record['valid'] or record.get('cleanup_error_type'):
            report(directory)
            print('PENDING VALID BENCHMARK: ' + trial['id'])
            return
    report(directory)


def report(directory):
    plan = read(directory / 'plan.json')
    trials = [read(p) for p in sorted(directory.glob('*/trial.json'))]
    valid = [t for t in trials if t.get('valid') and t.get('fingerprint') == plan['fingerprint']]
    # Refuse comparisons when the runtime image or installed packages differ across arms.
    environments = {digest({'versions': t.get('versions'), 'gpu': t.get('gpu'), 'python': t.get('python')}) for t in valid}
    comparable = len(environments) == 1
    summary = {'status': 'PENDING VALID BENCHMARK', 'hypothesis': plan['hypothesis'],
               'attempts': len(trials),
               'invalid': sum(not t.get('valid') and 'attempt_elapsed' in t for t in trials),
               'in_progress': sum(not t.get('valid') and 'attempt_elapsed' not in t for t in trials),
               'arms': {},
               'raw_trial_index': [str(p.relative_to(directory)) for p in sorted(directory.glob('*/trial.json'))]}
    for arm in ('A', 'B'):
        rows = [t for t in valid if t['arm'] == arm]
        summary['arms'][arm] = {'valid_trials': len(rows)}
        for key in ('e2e', 'model_source_prepare', 'gpu_session_minutes_observed', 'rtf', 'merge', 'cleanup'):
            values = [r[key] for r in rows if key in r]
            summary['arms'][arm][key] = {'individual': values, 'median': statistics.median(values) if values else None,
                                        'range': [min(values), max(values)] if values else None}
    if comparable and all(summary['arms'][a]['valid_trials'] >= 3 for a in ('A', 'B')):
        baseline, candidate = [summary['arms'][a]['e2e']['median'] for a in ('A', 'B')]
        summary.update(speedup=baseline/candidate, latency_reduction_percent=100*(1-candidate/baseline),
                       status='INCONCLUSIVE', note='Performance measured; human quality and boundary review required before PROMISING.')
    if not comparable:
        summary['limitation'] = 'No matched runtime environment across valid trials'
    save(directory / 'summary.json', summary)
    with (directory / 'trials.csv').open('w', newline='', encoding='utf-8') as stream:
        fields = ['id', 'arm', 'valid', 'e2e', 'model_source_prepare', 'gpu_session_minutes_observed', 'rtf']
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(trials)
    # Output equality is explicitly distinct from accuracy against corrected reference text.
    comparisons = []
    for trial in trials:
        if trial['arm'] != 'B' or not trial.get('valid'):
            continue
        a = directory / f"{trial['repetition']:02}-A"
        b = directory / trial['id']
        if (a / 'segments.json').is_file():
            baseline = read(a/'trial.json')
            if not baseline.get('valid'):
                continue
            comparison = {'candidate': trial['id'], **compare_segments(read(a/'segments.json'), read(b/'segments.json')),
                'txt_exact': sha(a/'transcript.txt') == sha(b/'transcript.txt'),
                'srt_exact': sha(a/'transcript.srt') == sha(b/'transcript.srt')}
            if trial.get('quality') and baseline.get('quality'):
                comparison['delta_percentage_points'] = {key: trial['quality'][key] - baseline['quality'][key]
                    for key in ('wer_percent', 'cer_percent')
                    if trial['quality'][key] is not None and baseline['quality'][key] is not None}
            comparisons.append(comparison)
    save(directory / 'comparisons.json', comparisons)
    audits = []
    for trial in valid:
        if plan['hypothesis'] != 'H2' or trial['arm'] != 'B':
            continue
        path = directory / trial['id'] / 'segments.json'
        if path.is_file():
            audit = read(path.with_name('boundary-audit.json'))
            audits.append({'trial': trial['id'], 'audio_reference_review': 'PENDING',
                           'issues': boundary_issues(read(path), audit.get('seam', plan['freeze']['duration']/2))})
    save(directory / 'boundary-findings.json', audits)
    lines = ['# Kaggle optimization experiment', '', '**' + summary['status'] + '**', '',
             '| Arm | Valid trials | Median model preparation (s, sum of jobs) | Median E2E (s) | Median GPU minutes (observed) |',
             '| --- | ---: | ---: | ---: | ---: |']
    for arm, values in summary['arms'].items():
        lines.append(f"| {arm} | {values['valid_trials']} | {values['model_source_prepare']['median']} | {values['e2e']['median']} | {values['gpu_session_minutes_observed']['median']} |")
    lines += ['', 'Raw trials:'] + ['- ' + p for p in summary['raw_trial_index']]
    lines += ['', 'Individual values and ranges: summary.json; flat timings: trials.csv.',
              'Quality: each trial.json contains reference WER/CER when supplied. comparisons.json reports output equality.',
              'H2 boundary-audit.json requires human review. Equality with baseline is not ground truth.',
              'boundary-findings.json flags overlapping repeated phrases without modifying measured outputs.',
              'Queue/running timestamps are polling observations (10-second interval), not platform billing.',
              'E2E starts before copying/uploading audio and ends when verified merged outputs are written. Cleanup is separate.']
    (directory / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(summary['status'])


def append_trial(directory, arm):
    """Add an explicitly requested replacement with a new ID; preserve all old evidence."""
    plan = read(directory / 'plan.json')
    ledger = read(directory / 'ownership.json')
    if any(not r.get('deleted') for r in ledger['resources']):
        raise ValueError('Finish cleanup before extending an experiment')
    repetition = max(t['repetition'] for t in plan['trials']) + 1
    if repetition > 99:
        raise ValueError('Experiment trial limit reached')
    trial = {'id': f'{repetition:02}-{arm}', 'arm': arm, 'repetition': repetition,
             'added_at': time.time(), 'reason': 'Explicit additional repetition; previous evidence retained'}
    plan['trials'].append(trial)
    save(directory / 'plan.json', plan)
    print(trial['id'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--audio', type=Path, required=True)
    prep.add_argument('--duration', type=float, required=True, help='Exact decoded duration in seconds')
    prep.add_argument('--owner', required=True)
    prep.add_argument('--hypothesis', choices=['H1', 'H2'], required=True)
    prep.add_argument('--model-dataset')
    prep.add_argument('--overlap', type=int, choices=[0, 5, 10], default=5)
    prep.add_argument('--split-strategy', choices=['midpoint', 'chunk-aligned'], default='midpoint')
    prep.add_argument('--repeats', type=int, default=3)
    prep.add_argument('--output', type=Path, required=True)
    for name in ('run', 'report', 'cleanup'):
        sub = commands.add_parser(name)
        sub.add_argument('directory', type=Path)
        if name != 'report':
            sub.add_argument('--cli-root', type=Path, required=True, help='Existing Chep data root containing tools/kaggle')
        if name == 'run':
            sub.add_argument('--gpu-eligibility-confirmed', action='store_true')
            sub.add_argument('--reference', type=Path, help='Manually corrected UTF-8 reference')
            sub.add_argument('--resume', action='store_true', help='Skip terminal cleaned attempts, including failures; no resubmission')
    extra = commands.add_parser('append')
    extra.add_argument('directory', type=Path)
    extra.add_argument('--arm', choices=['A', 'B'], required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        if args.repeats < 3 or not re.fullmatch(r'[a-zA-Z0-9_-]+', args.owner):
            parser.error('Need >=3 repeats and valid owner')
        prepare(args)
    elif args.command == 'run':
        execute(args)
    elif args.command == 'report':
        report(args.directory)
    elif args.command == 'append':
        append_trial(args.directory, args.arm)
    else:
        from local_app.kaggle_cli import KaggleCLI
        cleanup(args.directory, KaggleCLI(args.cli_root))


if __name__ == '__main__':
    main()
