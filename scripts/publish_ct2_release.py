"""Verify benchmark equivalence, publish via authenticated hf CLI, pin public artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

from huggingface_hub import HfApi, hf_hub_url

ROOT = Path(__file__).resolve().parents[1]
REPOS = {'es': 'nunu-willump/whisper-large-v3-turbo-es-ct2',
         'vi': 'nunu-willump/PhoWhisper-medium-ct2'}
FILES = ('config.json', 'model.bin', 'tokenizer.json', 'preprocessor_config.json', 'vocabulary.json')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--hf', type=Path, required=True)
    args = parser.parse_args()
    catalog_path = ROOT / 'config/transcription_models.json'
    catalog = json.loads(catalog_path.read_text(encoding='utf-8'))
    evidence = {}
    # Validate both folders before any publication.
    for code, repo in REPOS.items():
        folder = args.artifacts / code
        expected = json.loads((ROOT / f'docs/evidence/kaggle-profiles/{code}-artifact.json').read_text())
        provenance = json.loads((folder / 'conversion.json').read_text())
        assert provenance['matches_benchmarked_artifact'] is True
        assert provenance['source_revision'] == expected['revision']
        manifest = {}
        for name in FILES:
            path = folder / name
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            meta = {'size': path.stat().st_size, 'sha256': digest}
            assert meta == expected['files'][name], (code, name)
            assert meta == provenance['files'][name], (code, name)
            manifest[name] = meta
        evidence[code] = {'repo': repo, 'source_revision': expected['revision'], 'files': manifest}
    public = HfApi(token=False)
    for code, result in evidence.items():
        repo = result['repo']
        subprocess.run([str(args.hf), 'upload', repo, str(args.artifacts / code), '.',
                        '--no-private', '--commit-message', 'Publish benchmark-verified FP16 CTranslate2 conversion'], check=True)
        info = public.model_info(repo, files_metadata=True)
        assert info.private is False and len(info.sha) == 40
        result['revision'] = info.sha
        remote = {f.rfilename: f for f in info.siblings}
        for name, meta in result['files'].items():
            item = remote[name]
            assert item.size == meta['size'], (code, name, 'remote size')
            url = hf_hub_url(repo, name, revision=info.sha)
            if item.lfs:
                assert item.lfs.sha256 == meta['sha256'], (code, name, 'remote LFS hash')
                request = urllib.request.Request(url, headers={'Range': 'bytes=0-1023'})
                with urllib.request.urlopen(request, timeout=60) as response:
                    with (args.artifacts / code / name).open('rb') as local:
                        assert response.read(1024) == local.read(1024), (code, name, 'public download')
            else:
                with urllib.request.urlopen(url, timeout=60) as response:
                    assert hashlib.sha256(response.read()).hexdigest() == meta['sha256']
            meta['url'] = url
        cfg = catalog['languages'][code]
        cfg.update(ct2_model=repo, revision=info.sha, requires_conversion=False,
                   legacy_ct2_revisions=[result['source_revision']], files=result['files'])
    (ROOT / 'docs/evidence/v3-huggingface.json').write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf-8')
    catalog_path.write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({code: {'repo': r['repo'], 'revision': r['revision']} for code, r in evidence.items()}))


if __name__ == '__main__':
    main()
