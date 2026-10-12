# Third-party components

This application uses Python (PSF license), React (MIT), Lucide (ISC), FastAPI (MIT), Uvicorn (BSD), faster-whisper (MIT), CTranslate2 (MIT), PyAV (BSD), ONNX Runtime (MIT), and their dependencies. Package metadata and license files are retained inside `runtime/Lib/site-packages`. Python's license is included with its embeddable distribution. `runtime-dependencies.json` records resolved versions and downloaded archive hashes.

Websockets (BSD) is included for authenticated browser connection tracking. The Windows close dialog uses the operating system's Common Controls. Optional Kaggle CLI (Apache-2.0) and its dependencies download separately; their wheel metadata and license files are retained in the data folder. The installer contains no Kaggle credentials or preinstalled Kaggle CLI.

GPU libraries are not bundled. On explicit request, the application downloads NVIDIA cuBLAS, CUDA NVRTC and cuDNN wheels from their official PyPI distributions, under NVIDIA's applicable license agreements. License files are retained with the extracted libraries in the data folder. No GPU driver is supplied. The unused cuDNN dispatch DLL from the CTranslate2 wheel is stripped from the CPU core and supplied by the optional runtime instead. Microsoft Visual C++ Redistributable is supplied unchanged as Microsoft's signed installer under its redistribution terms.

Models are downloaded only after the user selects them in Optional Components during Setup or requests them in the app. Source IDs, immutable revisions and SHA-256 hashes are recorded in `config/transcription_models.json`.

Spanish and Vietnamese are distributed as preconverted FP16 CTranslate2 artifacts at https://huggingface.co/nunu-willump/whisper-large-v3-turbo-es-ct2 and https://huggingface.co/nunu-willump/PhoWhisper-medium-ct2. Their repositories retain upstream model cards, license notices and conversion provenance. Conversion tools are used only during release preparation; PyTorch, Transformers and the conversion runtime are not shipped with this application.

Configured model sources:

French, English and German CT2 preservation mirrors are hosted at `nunu-willump/whisper-large-v3-french-distil-dec16-ct2`, `nunu-willump/distil-large-v3.5-ct2` and `nunu-willump/whisper-large-v3-turbo-german-ct2` on Hugging Face. Their weights are unchanged. Upstream cards, license notices and source provenance are retained in each repository.

- French: Bofeng Huang, MIT, https://huggingface.co/bofenghuang/whisper-large-v3-french-distil-dec16
- English: Distil-Whisper, MIT, https://huggingface.co/distil-whisper/distil-large-v3.5-ct2
- Spanish: adriszmar, MIT, https://huggingface.co/adriszmar/whisper-large-v3-turbo-es
- German: primeLine, Apache-2.0, https://huggingface.co/primeline/whisper-large-v3-turbo-german; CT2 conversion https://huggingface.co/ostwilkens/whisper-large-v3-turbo-german-ct2
- Vietnamese: VinAI, BSD-3-Clause, https://huggingface.co/vinai/PhoWhisper-medium. PhoWhisper: Automatic Speech Recognition for Vietnamese, Thanh-Thien Le, Linh The Nguyen and Dat Quoc Nguyen, ICLR 2024 Tiny Papers.

Original French artifact provenance:

- Author: Bofeng Huang
- Repository: https://huggingface.co/bofenghuang/whisper-large-v3-french-distil-dec16
- Revision: `16bd02185bdaa6b00fb4b0deb46e47ac1b754b8e`
- Upstream model card: MIT
- Model SHA-256: `3fb4a2ac35540f921ed51804edd19d6b8f02cfc45956472fb6635965a339da5c`

The Windows installer is produced with Inno Setup. Compiler licensing is separate from the generated installer; see https://jrsoftware.org/isinfo.php for commercial build use.
