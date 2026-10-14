"""Versioned model policy. All model-specific settings live in JSON."""
from copy import deepcopy
import json
import re
from .config import ROOT

CATALOG_PATH = ROOT / 'config' / 'transcription_models.json'

def load_catalog(path=CATALOG_PATH):
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('schema_version') != 1:
        raise ValueError('Unsupported model catalog version.')
    for language, cfg in data['languages'].items():
        if cfg['backend'] != 'ctranslate2' or not 1 <= cfg['chunk_seconds'] <= 60:
            raise ValueError(f'Invalid model configuration: {language}')
        profiles = [cfg.get('sweet_spot'), *cfg['fallback_order'], *cfg['cpu_fallback_order'], *cfg['provisional_gpu_order']]
        for profile in filter(None, profiles):
            if profile['concurrency'] not in (1, 2, 4, 8) or not profile['compute_type']:
                raise ValueError(f'Invalid execution profile: {language}')
        for name, meta in cfg['files'].items():
            if '/' in name or '\\' in name or name in ('.', '..') or meta['size'] <= 0 or len(meta['sha256']) != 64:
                raise ValueError(f'Invalid artifact manifest: {language}')
        if not re.fullmatch(r'[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+/[1-9][0-9]*', cfg.get('kaggle_dataset', '')):
            raise ValueError(f'Invalid public Kaggle model Dataset: {language}')
    return data['languages']

LANGUAGES = load_catalog()

def model_config(language):
    if language not in LANGUAGES:
        raise ValueError('Select one of the supported languages.')
    return deepcopy(LANGUAGES[language])

def selection(language, device):
    return {'language': language, 'device_preference': device, 'model_config': model_config(language)}

def resolve_artifact(language, config):
    """Upgrade legacy conversion artifacts without changing frozen job settings."""
    result = deepcopy(config)
    current = model_config(language)
    if not result.get('requires_conversion'):
        if (result.get('revision') in current.get('legacy_ct2_revisions', []) and
                result.get('model') == current['model'] and
                {n:(m['size'],m['sha256']) for n,m in result['files'].items()} ==
                {n:(m['size'],m['sha256']) for n,m in current['files'].items()}):
            for key in ('files','revision','ct2_model','legacy_ct2_revisions','kaggle_dataset'):
                result[key] = deepcopy(current[key])
        if (result.get('revision') == current['revision'] and
                {n:(m['size'],m['sha256']) for n,m in result['files'].items()} ==
                {n:(m['size'],m['sha256']) for n,m in current['files'].items()}):
            result['kaggle_dataset'] = current['kaggle_dataset']
        return result
    if (current.get('requires_conversion') or not current.get('ct2_model') or
            result.get('model') != current['model'] or
            result.get('source_revision', result['revision']) != current.get('source_revision') or
            result['revision'] not in current.get('legacy_ct2_revisions', [])):
        raise ValueError('This legacy model needs a compatible preconverted release. Update the application.')
    for key in ('files', 'revision', 'ct2_model', 'requires_conversion', 'legacy_ct2_revisions', 'kaggle_dataset'):
        result[key] = deepcopy(current[key])
    return result

def public_catalog():
    return [{'language': code, **{k: cfg[k] for k in ('label', 'model', 'ct2_model', 'backend', 'chunk_seconds',
            'benchmark_status', 'validation_status', 'requires_conversion', 'license')}} for code, cfg in LANGUAGES.items()]

def is_validated(selected):
    """Allow old queued selections once the identical artifact/settings are verified."""
    cfg = selected['model_config']
    if cfg['validation_status'] == 'validated':
        return True
    current = LANGUAGES.get(selected['language'], {})
    return current.get('validation_status') == 'validated' and all(
        current.get(key) == cfg.get(key) for key in ('model','revision','files','chunk_seconds','transcribe'))
