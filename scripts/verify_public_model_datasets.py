"""Verify every pinned public Kaggle model Dataset and its artifact file sizes."""
from __future__ import annotations

import argparse
import csv
import io
import json
import urllib.request
from pathlib import Path

from local_app.catalog import LANGUAGES
from local_app.kaggle_cli import KaggleCLI


def rows(output):
    return list(csv.DictReader(io.StringIO('\n'.join(line for line in output.splitlines() if line.strip()))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cli-root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    cli = KaggleCLI(args.cli_root)
    owner = 'longnj'
    listing, _ = cli.run(['datasets', 'list', '--user', owner, '--csv'], timeout=120)
    public = {row['ref'] for row in rows(listing)}
    results = {}
    for language, cfg in LANGUAGES.items():
        versioned = cfg['kaggle_dataset']
        ref, version = versioned.rsplit('/', 1)
        if ref not in public:
            raise ValueError(f'Not publicly listed: {versioned}')
        if version != '1':
            raise ValueError(f'Unexpected pinned Dataset version: {versioned}')
        status, _ = cli.run(['datasets', 'status', ref], timeout=120)
        if status.strip().lower() != 'ready':
            raise ValueError(f'Not ready: {versioned}')
        listing, _ = cli.run(['datasets', 'files', ref, '--csv'], timeout=120)
        files = {row['name']: int(row['size']) for row in rows(listing)}
        for name, meta in cfg['files'].items():
            if files.get(name) != meta['size']:
                raise ValueError(f'Artifact size mismatch: {versioned}/{name}')
        url = 'https://www.kaggle.com/api/v1/datasets/view/' + ref
        request = urllib.request.Request(url, headers={'User-Agent': 'Chep public-data check'})
        with urllib.request.urlopen(request, timeout=30) as response:
            anonymous = json.load(response)
        if (anonymous.get('ref') != ref or anonymous.get('isPrivate') is not False or
                anonymous.get('currentVersionNumber') != 1):
            raise ValueError(f'Anonymous access or version check failed: {versioned}')
        results[language] = {'dataset': versioned, 'public': True,
                             'anonymous_http': 200, 'files_verified': len(cfg['files'])}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
