"""Make lossless prefix clips from a decoded benchmark WAV for H2 duration trials."""

import argparse
import hashlib
import json
import wave
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--minutes', type=int, nargs='+', default=[12, 21])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lengths = sorted(set(args.minutes))
    with wave.open(str(args.source), 'rb') as source:
        if source.getnchannels() != 1 or source.getsampwidth() != 2 or source.getframerate() != 16000:
            raise ValueError('Expected mono PCM16 at 16000 Hz')
        if lengths[-1] * 60 * 16000 > source.getnframes():
            raise ValueError('Source is shorter than requested duration')
        manifest = {'source': str(args.source.resolve()), 'source_sha256': digest(args.source),
                    'clips': []}
        for minutes in lengths:
            source.rewind()
            target = args.output / f'prefix-{minutes}m.wav'
            remaining = minutes * 60 * 16000
            with wave.open(str(target), 'wb') as clip:
                clip.setparams(source.getparams())
                while remaining:
                    frames = source.readframes(min(remaining, 16000 * 60))
                    if not frames:
                        raise RuntimeError('Source ended early')
                    clip.writeframesraw(frames)
                    remaining -= len(frames) // 2
            manifest['clips'].append({'minutes': minutes, 'seconds': minutes * 60,
                                      'path': str(target.resolve()), 'sha256': digest(target),
                                      'bytes': target.stat().st_size})
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
