# Kaggle optimization benchmarks and integration

Benchmarks were run on 2026-10-14. The production Kaggle path was subsequently
updated at the user's request; the historical benchmark method and results below
remain unchanged. No installer or deployment is included in this record.

## Current result

| Arm | Valid trials | Model preparation (s) | E2E (s) | GPU minutes | WER/CER | Conclusion |
| --- | ---: | --- | --- | --- | --- | --- |
| H1-A: HF | 1 | 114.842 | 595.964 | 9.061 | No reference | PENDING VALID BENCHMARK |
| H1-B: public Dataset | 1 | 26.195 | 489.846 | 7.478 | No reference | PENDING VALID BENCHMARK |
| H2-A: whole 99-minute recording | 3 | 20.488 median | 482.989 median | 7.179 median | No reference | Latency baseline measured |
| H2-B: two notebooks, 5-second overlap | 3 | 53.524 median (sum) | 350.099 median | 9.346 median (sum) | No reference | 27.51% median E2E reduction |
| H2-A: 12-minute WAV | 3 | 23.107 median | 164.696 median | 2.312 median | No reference | Latency baseline measured |
| H2-B: 12-minute WAV, 5-second overlap | 1 valid of 4 completed | 62.192 operational median (sum) | 185.299 operational median of all 4 | 4.337 operational median (sum) | No reference | Does not meet 20% gate |
| H2-A: 21-minute WAV | 3 | 21.364 median | 188.322 median | 2.633 median | No reference | Latency baseline measured |
| H2-B: 21-minute WAV, 5-second overlap | 2 valid of 3 completed | 39.750 operational median (sum) | 158.056 operational median of all 3 | 3.801 operational median (sum) | No reference | 16.07% reduction; user accepts this observed gain |

These H1 entries are **individual observations, not established repeated-run medians**.
The first pair has exactly equal 433 structured segments, timestamps, TXT and SRT,
with identical dependency inventories, Python and GPU class. This is output equivalence,
not ground-truth accuracy. No reference WER/CER is available.
H1 attempt `02-B` lost controller/CLI connectivity; its remote output was recovered
and all owned resources were cleaned up. Its interrupted E2E is excluded even though
the notebook completed. A distinct replacement `04-B` was planned but **will not run**:
the user subsequently requested that H1 stop and only H2 continue. All H1 remote
resources have been deleted; old evidence remains.
The 99-minute H2 experiment has three valid measurements per arm. Baseline E2E values
are 513.440, 482.989 and 453.141 seconds; dual values are 350.099, 327.897 and
351.455 seconds. The ratio of medians is 1.380x, a 27.51% reduction, above the
user's 20% latency criterion **at this duration**. Dual mode used more observed GPU
time (9.346 versus 7.179 GPU-session minutes by median). Two of three seam audits
flagged the same nine-word repeated phrase. The user prefers retaining occasional
repeated words over automatic partial-phrase removal that could drop speech, so the
conservative merge remains in place. There is no corrected reference transcript;
quality and the shorter-duration decision remain inconclusive.

At 12 minutes, baseline E2E values were 148.119, 164.696 and 172.649 seconds.
All four dual attempts completed and produced outputs: 166.072, 183.332, 200.719
and 187.266 seconds. Only the first had observed inference overlap; the other three
were marked invalid for a speedup claim, because their short inference phases did not
overlap despite concurrent submission. Their operational E2E still counts as evidence
about admission/startup risk. The median of all completed dual attempts was 185.299
seconds, 12.51% slower than the baseline median. This is **not** a valid three-pair
speedup estimate; the observed data do not support dual mode at 12 minutes.
The user then requested skipping 15 and 18 minutes and measuring 21 minutes next.
At 21 minutes, baseline E2E values were 209.191, 174.404 and 188.322 seconds;
dual values were 153.611, 158.056 and 176.176 seconds. The last dual attempt
completed without inference overlap and is invalid for a repeated-speedup claim.
Including all completed attempts, the operational median reduction was 16.07%,
short of the originally specified 20% gate. The first pair alone had suggested
26.6%; the three-attempt series shows why the criterion must use repeated E2E.
The user explicitly judged the observed 16.1% worthwhile and asked to stop.
Thus 21 minutes is a **candidate duration under the user's revised judgment**, not
a demonstrated 20%-gate threshold: only two dual runs meet the strict inference-
overlap validity rule, there is no corrected quality reference, and 15/18 minutes
were not run. At the time this benchmark stopped, no production threshold or
dual-mode behavior had been changed; the later integration is recorded below.
The two valid 21-minute dual outputs have no automatically flagged repeated phrase
at the 5-second seam; this is not a manual audio-reference quality check.

The user suspected that a 30-second delay in receiving the second GPU could erase
the short-file inference saving. The measured **RUNNING-observation skew** between
the two notebooks was only 1.8–2.5 seconds in these seven completed short-file dual
attempts. The **first-inference skew** was much larger in the non-overlapping cases:
28.36, 52.05 and 40.15 seconds at 12 minutes, and 34.33 seconds at 21 minutes.
Thus the observed failure mode here was differing time to inference readiness,
including dependency/model startup, rather than a measured 30-second difference in
GPU RUNNING allocation. Status polling is approximate; these are observations from
this campaign, not a general claim about Kaggle scheduling.

The 30-minute plan had already started when the stop request arrived. Its first
baseline notebook was cancelled before output; its private notebook and audio
Dataset were deleted. It contributes **no latency result**. The 15/18-minute plans
were prepared but never submitted. No further Kaggle measurements are scheduled.

Local raw-trial indexes (private outputs remain gitignored):

- `test-results/kaggle-optimization/preflight-fixed/preflight.json`: T4 allocation.
- `test-results/kaggle-optimization/h1-longnj/plan.json`: frozen H1 configuration.
- `test-results/kaggle-optimization/h1-longnj/01-A/trial.json`: valid first HF trial.
- `test-results/kaggle-optimization/h1-longnj/01-B/trial.json`: valid first Dataset trial.
- `test-results/kaggle-optimization/h1-longnj/02-B/trial.json`: excluded interrupted attempt.
- `test-results/kaggle-optimization/h1-longnj/02-B/recovered-output/result.json`: recovered notebook evidence only.
- `test-results/kaggle-optimization/h2-longnj/plan.json`: independent H2 configuration.
- `test-results/kaggle-optimization/h2-longnj/summary.json`: all six valid 99-minute H2 trials.
- `test-results/kaggle-optimization/h2-longnj/boundary-findings.json`: partial-repeat audit; measured outputs remain unchanged.
- `test-results/kaggle-optimization/h2-12m-longnj/summary.json`: valid-trial count and 12-minute baseline.
- `test-results/kaggle-optimization/h2-12m-longnj/*/trial.json`: all four completed dual attempts, including three without inference overlap.
- `test-results/kaggle-optimization/h2-21m-longnj/summary.json`: 21-minute valid-trial count and baseline.
- `test-results/kaggle-optimization/h2-21m-longnj/*/trial.json`: all three completed dual attempts, including the non-overlapping attempt.
- `test-results/kaggle-optimization/h2-30m-longnj/01-A/trial.json`: user-cancelled partial attempt, excluded from latency analysis.
- `test-results/kaggle-optimization/h2-30m-longnj/ownership.json`: both 30-minute resources marked deleted.
- `test-results/kaggle-optimization/threshold-audio/manifest.json`: lossless clip hashes and durations.

The authenticated account was confirmed as `longnj`. The API listed
`longnj/french-whisper-ct2` as public, current version **1**. The corrected private
preflight successfully obtained **Tesla T4**. Two earlier preflights failed because
the probe lacked kernelspec metadata (a harness bug, fixed); these are excluded.
The user explicitly approved uploading the recording privately to Kaggle. The retained local
copy is `benchmarks/kaggle_french_real_trial/hugging_face/work/23c68071226344d0a678803b1c92061c/input/23c68071226344d0a678803b1c92061c.audio`;
the original name is recorded in the prior trial document, not the retained filename.
Its SHA-256 is `165bd973e35710fbbc1a4528287021190a6fbd465a6dc8930a794d725e2aad72`,
size 97,159,094 bytes, decoded duration 5,955.5626875 seconds.
Failure elapsed times are not performance measurements. See local raw evidence under
`test-results/kaggle-optimization/`; no credentials are included in generated notebooks.

All observations from `benchmarks/kaggle_dataset_french_trial.md` are excluded from
performance tables and acceptance checks. Its old completed/failure E2E times are
confounded; its preparation figures are single-run observations, not evidence of
repeatable speedup. Old TXT/SRT equality is only a preliminary functional check.

## Frozen baseline

- `local_app/kaggle_backend.py:KaggleBackend.run`: private audio Dataset, kernel
  submission with `--accelerator NvidiaTeslaT4`, status polling and verified output
  retrieval. Existing cleanup deletes recorded owned resources; cancellation checks
  stop subprocesses/polling. The experiment implements its own lifecycle and ledger.
- `local_app/kaggle_notebook.py:notebook`: HF `snapshot_download`, pinned file sizes
  and SHA-256 checks, CUDA device 0, four CPU threads, production `worker.run`.
- `config/transcription_models.json:languages.fr`: CT2 repository
  `nunu-willump/whisper-large-v3-french-distil-dec16-ct2`, revision
  `b259ae49f9fdc9fac3ec293a0019f1d04646eb58`; FP16, concurrency 2, 40-second
  chunks, French, beam 5, VAD disabled, previous-text conditioning enabled.
- `local_app/worker.py:run`: same WhisperModel constructor, concurrent seeded noise
  warmup, streaming mono PCM16/16 kHz decode, two chunk workers, TXT/SRT formatting.
  Each `prepare` snapshots worker/config/hardware source, catalog, artifact manifest,
  input checksum and dependency pins into `plan.json`, with a SHA-256 fingerprint.
  Production currently has no Dataset model-source default.
- Baseline always uses one notebook and HF. No fallback or profile retries are
  permitted in the benchmark. Each run rechecks actual compute type and T4 identity.

## Reproducible execution

From the repository root, use Python 3.12 with the existing authenticated Chep
Kaggle CLI 2.2.4+. Local unit tests need pytest; analysis helpers use the standard
library. Remote dependencies come from the same frozen inference pins extracted
from `requirements-lock.txt` in both arms. All installed versions and Python version
are recorded; differing inventories block a speedup report. An unavailable pin
invalidates the run. Offline wheels are not part of this experiment.

```powershell
$cliRoot = "$env:LOCALAPPDATA\FrenchTranscriptLocal\data"
python -m scripts.kaggle_optimization.cli report test-results/kaggle-optimization/h2-21m-longnj
python -m scripts.kaggle_optimization.cli report test-results/kaggle-optimization/h2-12m-longnj
python -m pytest -q -p no:cacheprovider tests/test_kaggle_optimization.py tests/test_kaggle.py tests/test_kaggle_gpu.py
```

To reproduce an H2 run in a **new, explicitly authorized experiment**, first generate
lossless clips with `python -m scripts.kaggle_optimization.threshold_clips --source
test-results/kaggle-optimization/decoded.wav --output
test-results/kaggle-optimization/threshold-audio --minutes 12 21`. Then use
`cli prepare --hypothesis H2 --overlap 5 --repeats 3 --audio <clip> --duration
<exact seconds> --owner longnj --output <new directory>` and `cli run <new directory>
--cli-root $cliRoot --gpu-eligibility-confirmed`. This documents reproduction; the
current campaign is stopped.

Only pass `--gpu-eligibility-confirmed` after account verification/quota checks and
a successful GPU preflight. Every measured job still independently checks T4.
Add `--reference C:\path\corrected.txt` to `run` for ground-truth WER/CER.
The recording is uploaded privately for each trial. The default is three repetitions
per arm in alternating AB/BA/AB order. Each invocation owns a unique experiment
prefix. Reusing an attempted experiment fails rather than resubmitting ambiguously.
An invalid run stops the campaign; inspect evidence and finish cleanup. `run --resume`
skips terminal cleaned attempts (including failures) and runs only previously
unattempted trials. It never resubmits an old ID. `append DIRECTORY --arm B` (or A)
adds an explicitly requested extra repetition with a new ID and timestamp while
preserving the original frozen configuration and all failed evidence. Otherwise,
create a fresh experiment. Cleanup is attempted after both successful and failed trials. Ctrl+C
records an interrupted attempt; use `cleanup` afterward if cleanup was interrupted.
Cleanup checks authenticated owner and every exact ledger ref, never enumerates
other user resources, and can be retried after a failure.

Kaggle source strings support `owner/slug/version-number`; the generator requires
the French Dataset version explicitly. See the official
[Kaggle CLI source validation](https://github.com/Kaggle/kaggle-api/blob/main/src/kaggle/api/kaggle_api_extended.py).
Runtime discovery recursively searches `/kaggle/input`, verifies the full pinned
manifest, rejects ambiguous matches and loads directly without copying model files.

## Matrix and measurement boundaries

H1 was stopped at the user's request. The H2 threshold series uses real-speech
prefixes of 12 and 21 minutes from the approved recording, each as a separate
experiment with exactly 5 seconds of overlap. The prepared 15/18-minute plans were
not submitted, following the user's revised instruction; the 30-minute run was
cancelled before output. The clips are lossless
mono PCM16/16 kHz WAV extracted from the previously decoded source; this changes
the upload/decoder workload relative to the original compressed 99-minute input.
Within each duration, both arms use the identical clip. Do not pool unlike
durations or extrapolate a result to other audio formats. No zero-overlap control
will run. H1+H2 combined work is outside this benchmark.

H2 uses one original recording Dataset per trial, mounted by both jobs. Each job
decodes the original using the unchanged production decoder and selects a lossless
PCM range. This deliberately conservative implementation charges full decoding
twice; it does **not** claim compressed-audio seek performance. Separate shard
uploads and silence-aware planning remain optional unimplemented experiments.
The total shared overlap is 5 seconds, centered on the midpoint.
Internal chunk concurrency stays at two per GPU.
Each shard restarts the worker's chunk grid at its own range origin. The 40-second
chunk length and decoding settings remain unchanged, but later chunk boundaries
can differ from the whole-file baseline. Evaluate the entire transcript as well as
the seam; this experiment does not isolate seam effects from chunk-context changes.

An earlier zero-overlap chunk-aligned plan exists only as a local, unused plan. The
user explicitly chose 5-second overlap to preserve content around difficult seams;
that zero-overlap plan will not be submitted.

`trial.json` records controller timestamps, status history, attempts, upload/readiness,
retrieval, merge, E2E, cleanup, RTF and observed GPU-session minutes. Each job's
`output/result.json` records GPU inventory, environment, full-precision segments,
artifact hashes and notebook phase timings:

- Scheduling: submit to first observed RUNNING; never assigned to model preparation.
- Dependency readiness: runtime entry through GPU probe and pip install. The small
  notebook bootstrap and host provisioning are outside this boundary and inside E2E.
- Model preparation: HF snapshot acquisition plus integrity verification, or mounted
  directory discovery plus verification. No model copy. Fresh cache directory per
  notebook; platform caches are uncontrolled and no cold-start claim is made.
- CT2 load/warmup: production LOADING to validated event.
- Decode/range: production streaming decode plus optional PCM range extraction.
- Inference: wall time collecting each production two-chunk batch; UTC intervals
  are stored for every batch. Both H2 jobs must have overlapping intervals.
- Output write/loop overhead: transcription phase less measured batch execution.
  This includes formatting/I/O and loop bookkeeping, not a pure filesystem timer.
- E2E: before local copy/upload to verified TXT/SRT (merged for H2) being available
  locally. Account lookup and one-time plan creation are excluded. Cleanup is a
  separate post-output metric. `attempt_elapsed` is retained even on failure.
- GPU-session minutes: summed observed RUNNING→COMPLETE intervals / 60. Polling is
  every 10 seconds, so these are estimates, not Kaggle billing/quota API readings.
  Each job uses GPU index 0; extra devices allocated by Kaggle are not exploited.

UTC window overlap assumes notebook clocks are adequately synchronized. Very small
overlaps require scrutiny. Scheduling/quota failures invalidate H2; sequential jobs
cannot produce a dual-GPU speedup claim. There are no automatic retries.

## Output and quality analysis

Each report writes `summary.json` (individual values, medians, ranges), `trials.csv`,
`comparisons.json` (exact structured segments/TXT/SRT equality), and `report.md`.
Reference WER/CER use NFC, case folding, punctuation-to-space and whitespace
normalization while preserving French accents. They are percentages; subtract the
baseline percentage from the candidate percentage for percentage-point deltas.
Reference files must match the entire recording and be manually corrected.

Merge rebases local segments by the shard's global offset, validates bounds and
sorts by timestamp. It removes only cross-shard equal normalized segments with
overlapping times and endpoints within two seconds. Repeated phrases elsewhere
are retained. Partial phrase matches are deliberately retained: they require human
review, and the harness does not guarantee automatic boundary repair. Inspect
`boundary-audit.json` and the original audio within ±30 seconds of the midpoint,
checking dropped/repeated words, timing drift and overlaps. Segment overlap is
permitted by SRT but may be undesirable; inspect it before any quality acceptance.

At least three matched valid repetitions in each arm are required for numerical
speedup. `speedup = median(A E2E) / median(B E2E)`; reduction is
`100 * (1 - median(B)/median(A))`. The originally specified latency gate for dual
mode was at least 20% median E2E reduction from the one-notebook baseline.
The user subsequently accepted the observed 16.1% operational reduction at 21
minutes and stopped the experiment. These are distinct decisions; accepting the
observation does not retroactively make it pass the 20% criterion. The 12/21-minute
measurements include actual startup and GPU allocation delays. The skipped
15/18-minute controls mean the minimum useful duration within that interval is
unknown. Automated reports remain INCONCLUSIVE until quality/boundary and reliability
review. Missing valid GPU data always means PENDING VALID BENCHMARK.

## Production integration

The current catalog pins one public model Dataset version per language. A Kaggle job
always creates its audio Dataset privately under the **signed-in user's** account,
then attaches both that private input and the versioned public model Dataset to a
private notebook. Model ownership is never used as the job owner. The notebook
discovers the mounted model and verifies every pinned file's size and SHA-256 before
loading it; a missing, inaccessible or modified Dataset fails explicitly instead of
silently falling back to a Hugging Face download. Kaggle states that
[anyone can access public Datasets](https://www.kaggle.com/docs/notebooks#collaborating-on-datasets),
including from notebooks owned by another account. Tests exercise that separation
with a job owner different from `longnj`.

The pinned public versions are `longnj/french-whisper-ct2/1`,
`longnj/english-whisper-ct2/1`, `longnj/spanish-whisper-ct2/1`,
`longnj/german-whisper-ct2/1`, and `longnj/vietnamese-whisper-ct2/1`.
The four non-French Datasets were published from SHA-256-verified model files.
The verification record at `test-results/model-publication/verification.json`
confirms that all five are publicly listed, ready, contain the pinned model
files, and expose anonymous HTTP metadata with `isPrivate=false` and version 1.
This establishes public visibility without using `longnj` credentials for the
HTTP check. It does not substitute for a live notebook run under a second
account; no second GPU-capable account was available.

The worker counts decoded 16 kHz samples before submission. Recordings **shorter
than 21 minutes** use one notebook; recordings **at least 21 minutes** use two.
The two ranges overlap by exactly five seconds around the midpoint. Each notebook
decodes the same private original, selects its own lossless PCM range and returns
structured segments plus verified TXT/SRT. Both notebook IDs, submission state and
cleanup state are persisted for restart, cancellation and retry. The private audio
Dataset is deleted only after both owned notebooks have been handled.

The production merge checks only the five-second seam. It removes a prefix from
the second shard only when at least four Unicode-normalized words exactly match a
suffix in an overlapping first-shard segment, their timestamps overlap by at least
one second, and there is a unique best match. Exact lexical agreement scores 1.0,
above the requested 0.90 threshold. This score is a deterministic matching rule,
**not** a calibrated probability. If evidence is weaker or ambiguous, both texts
are retained. The measured 99-minute seam's nine-word duplicate is removed by this
rule, while the continuation remains. The original benchmark transcripts were not
rewritten. No automatic word deletion is performed away from the seam.

The 21-minute threshold follows the user's explicit acceptance of the observed
16.1% operational E2E reduction; the earlier ≥20% gate was not met. Only two of
three 21-minute dual attempts had inference overlap, so this is not a three-valid-
pair benchmark claim. Separate Kaggle accounts still need their own authentication,
GPU eligibility and concurrent quota. Public Dataset access solves the model
ownership issue, not account-level GPU allocation limits.

Production-backend smoke runs on `longnj` used an approved local audio derivative
and the real private-audio/public-model attachment path. All produced verified
TXT/SRT output and finished with their private test resources cleaned:

| Language | Input duration | Route | Outcome |
| --- | ---: | --- | --- |
| French | 21 min | Two notebooks, 5 s overlap | Completed; both notebooks and private Dataset cleaned |
| English | 1 min | One notebook | Completed; resources cleaned |
| Spanish | 1 min | One notebook | Completed; resources cleaned |
| German | 1 min | One notebook | Completed; resources cleaned |
| Vietnamese | 1 min | One notebook | Completed; resources cleaned after a transient first cleanup failure and retry |

These runs verify attachment, model-file checksum checks, routing, output import,
and cleanup. The one-minute non-French runs used French source speech, so they
do **not** establish transcription quality in those languages. The 21-minute
French run had no seam words removed. The separate measured 99-minute seam's
nine-word duplicate was removed by the merge rule in a local replay. No live
second-account notebook run was possible; public visibility and the different-
owner code path were checked as described above.
