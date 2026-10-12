"""Release gate: import measured profiles only when all model/device checks pass."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]

def select_gpu(rows):
    successful = [r for r in rows if r.get('status') == 'ok' and r['profile']['device'] == 'cuda']
    baseline = next(r for r in successful if r['profile']['compute_type'] == 'float16' and r['profile']['concurrency'] == 1)
    baseline_wer = max(r['wer'] for r in baseline['runs'] if r.get('phase') == 'measured')
    baseline_cer = max(r['cer'] for r in baseline['runs'] if r.get('phase') == 'measured')
    eligible = []
    seen = set()
    for row in successful:
        profile = row['profile']
        key = (profile['compute_type'], profile['concurrency'])
        if key in seen:
            continue
        seen.add(key)
        runs = [r for r in row['runs'] if r.get('phase') == 'measured']
        if len(runs) < 3 or any(not math.isfinite(r['rtf']) or not r.get('text', '').strip() for r in runs):
            continue
        if max(r['wer'] for r in runs) > baseline_wer + .01 or max(r['cer'] for r in runs) > baseline_cer + .01:
            continue
        # Pilot policy: no failures, <=1 percentage-point quality regression,
        # and a measured VRAM footprint with at least 1 GiB headroom on the 15 GiB T4.
        if row['peak_vram_mib'] > 14336:
            continue
        eligible.append((statistics.mean(r['rtf'] for r in runs), row['peak_vram_mib'], profile))
    if not eligible:
        raise RuntimeError('No eligible measured GPU profile.')
    fastest = min(r[0] for r in eligible)
    near_fastest = [r for r in eligible if r[0] <= fastest * 1.05]
    best = min(near_fastest, key=lambda r:(r[1],r[2]['concurrency']))
    ordered = [best] + sorted([r for r in eligible if r != best], key=lambda r:r[0])
    return [{'compute_type': r[2]['compute_type'], 'concurrency':r[2]['concurrency']} for r in ordered]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu-results', nargs='+', type=Path, required=True)
    parser.add_argument('--gpu-results', type=Path, required=True)
    args = parser.parse_args()
    catalog_path = ROOT/'config'/'transcription_models.json'
    catalog = json.loads(catalog_path.read_text(encoding='utf-8'))
    cpu = {}
    for path in args.cpu_results:
        report = json.loads(path.read_text(encoding='utf-8'))
        for code, item in report['languages'].items():
            if item.get('results'):
                cpu[code] = item
    gpu = json.loads(args.gpu_results.read_text(encoding='utf-8'))
    for code, cfg in catalog['languages'].items():
        measured = gpu['languages'][code]
        for item in (cpu[code], measured):
            if item['model'] != cfg['model'] or item['revision'] != cfg['revision']:
                raise RuntimeError(f'{code}: benchmark source mismatch')
        cpu_rows = [r for r in cpu[code]['results'] if r.get('status') == 'ok' and r['profile']['device'] == 'cpu']
        if not {'int8_float32','float32'} <= {r['profile']['compute_type'] for r in cpu_rows}:
            raise RuntimeError(f'{code}: CPU validation incomplete')
        if any(len(r['runs']) < 2 or any(not run['text'].strip() for run in r['runs']) for r in cpu_rows):
            raise RuntimeError(f'{code}: incomplete CPU quality results')
        cpu_baseline = next(r for r in cpu_rows if r['profile']['compute_type'] == 'float32')
        cpu_wer = max(r['wer'] for r in cpu_baseline['runs'])
        cpu_cer = max(r['cer'] for r in cpu_baseline['runs'])
        acceptable_cpu = [r for r in cpu_rows if max(x['wer'] for x in r['runs']) <= cpu_wer + .01
                          and max(x['cer'] for x in r['runs']) <= cpu_cer + .01]
        cfg['cpu_fallback_order'] = [{'compute_type':r['profile']['compute_type'], 'concurrency':1}
                                     for r in sorted(acceptable_cpu, key=lambda r:r['profile']['compute_type'] == 'float32')]
        profiles = select_gpu(measured['results'])
        if code != 'fr':
            cfg['sweet_spot'], cfg['fallback_order'] = profiles[0], profiles[1:]
            cfg['benchmark_status'] = 'validated'
        cfg['validation_status'] = 'validated'
        cfg['benchmark'] = {
            'gpu':gpu['gpu'], 'date_utc':gpu['created_utc'], 'versions':gpu['versions'],
            'dataset':'google/fleurs test, first 6 samples per language with 0.5-second separators',
            'gpu_repeats':3, 'warmup_repeats':1, 'cpu_repeats':2,
            'scope':'Small T4 pilot; preferred among tested candidates, not a universal optimum or domain-quality certification.',
            'selection':'Mean RTF after warmup; <=1 pp WER/CER regression vs FP16x1; prefer lower VRAM within 5% of fastest.',
            'report':'docs/evidence/kaggle-profiles/benchmark-results.json',
            'report_sha256':hashlib.sha256(args.gpu_results.read_bytes()).hexdigest(),
            'cpu_profiles':[r['profile'] for r in cpu_rows],
            'cpu_quality_policy':'Only candidates within 1 pp WER/CER of CPU FP32 are enabled; INT8 first when eligible.',
        }
        print(code,cfg['sweet_spot'])
    catalog_path.write_text(json.dumps(catalog,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

if __name__ == '__main__':
    main()
