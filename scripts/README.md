# Scripts Directory Index

Developer, release, verification, smoke-testing, and benchmark utilities for Chep.

## `scripts/build/`
Windows distribution build and pinned dependency/catalog generators:
- `build_windows.py` — Build the standalone embedded Python runtime and compile `installer/Chep.iss`.
- `prepare_installer_components.py` — Generate `installer/component-sizes.iss` from `config/transcription_models.json`.
- `prepare_gpu_catalog.py` — Resolve and pin official NVIDIA runtime wheel hashes in `config/gpu_runtimes.json`.
- `prepare_kaggle_cli.py` — Resolve and pin Kaggle CLI/pip wheel hashes in `config/kaggle_cli.json`.

## `scripts/release/`
Hugging Face model mirroring, CTranslate2 conversion release, and Kaggle public model Dataset management:
- `mirror_models.py` — Mirror pinned CTranslate2 models to Hugging Face with upstream attribution.
- `prepare_ct2_release.py` — Generate the CPU-only Kaggle conversion notebook (`benchmarks/ct2_release/`).
- `publish_ct2_release.py` — Verify and upload converted Spanish/Vietnamese CTranslate2 artifacts to Hugging Face.
- `publish_model_datasets.py` — Stage and publish pinned public CTranslate2 Kaggle Datasets (default owner: `french-transcript`).
- `verify_public_model_datasets.py` — Verify all pinned public Kaggle Datasets (`status`, file sizes, and anonymous HTTP visibility).

## `scripts/checks/`
Targeted UI, download, runtime, and native Windows dialog checks:
- `check_connection_ui.py` — Playwright UI check for Kaggle connection states, model selection, upload, and Start behavior.
- `check_ct2_downloads.py` — Verify local model cache repair/download via `ModelManager`.
- `check_native_close.py` — Exercise the Windows native TaskDialog close prompt.
- `check_oauth_launch.py` — Verify Kaggle CLI OAuth browser launch interception under an empty profile.
- `check_optional_runtime.py` — Exercise the optional CUDA runtime download queue without GPU inference.
- `check_packaged_pip.py` — Verify isolated pip installation of the optional Kaggle CLI inside the packaged embedded runtime.

## `scripts/smoke/`
End-to-end smoke scripts (browser + server / live Kaggle):
- `smoke_exit_packaged.py` — Packaged runtime + browser + native close dialog lifecycle smoke test.
- `smoke_local.py` — Browser -> local API -> CPU inference smoke test.
- `smoke_kaggle.py` — Opt-in live Kaggle + browser end-to-end smoke test (uses Kaggle quota).
- `kaggle_production_smoke.py` — Isolated production-backend Kaggle smoke runner (uses Kaggle quota).

## `scripts/benchmarks/`
Benchmarking harnesses and release profile importers:
- `apply_benchmark_results.py` — Import measured benchmark profiles into `config/transcription_models.json`.
- `prepare_kaggle_benchmark.py` — Generate Kaggle multilingual/GPU benchmark notebooks.
- `kaggle_app_startup_benchmark.py` — Measure Kaggle `RUNNING`-to-first-live-log startup intervals.
- `kaggle_optimization/` — H1/H2 Kaggle optimization experiment CLI, runner, and analysis package (`python -m scripts.benchmarks.kaggle_optimization.cli`).
