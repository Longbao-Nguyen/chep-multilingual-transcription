"""CLI entrypoint to compile-check, run Kaggle evaluation waves, summarize, and clean up."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.benchmarks.full_eval.kaggle_runner import (  # noqa: E402
    KernelSpec,
    cleanup_ledger,
    run_kaggle,
    submit_and_wait_wave,
)
from scripts.benchmarks.full_eval.nb_fleurs import build_fleurs_wave_specs  # noqa: E402
from scripts.benchmarks.full_eval.nb_french_evolution import (  # noqa: E402
    build_fr_ct2_spec,
    build_fr_dual_specs,
    build_fr_pt_spec,
)
from scripts.benchmarks.full_eval.nb_longform import build_longform_wave_specs  # noqa: E402
from scripts.benchmarks.full_eval.summarize import write_full_summary  # noqa: E402


def compile_check_all() -> None:
    specs: list[KernelSpec] = [
        *build_fleurs_wave_specs(),
        *build_longform_wave_specs(),
        build_fr_pt_spec(),
        build_fr_ct2_spec(),
        *build_fr_dual_specs(),
    ]
    for sp in specs:
        compile(sp.code, f"<remote:{sp.wave}>", "exec")
        print(f"[compile ok] {sp.wave}: {len(sp.code)} chars, datasets={sp.dataset_sources}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Chep full Kaggle T4 evaluation waves.")
    parser.add_argument(
        "--step",
        required=True,
        choices=["check", "fleurs", "longform", "fr-wave3a", "fr-dual", "summarize", "cleanup", "all"],
    )
    args = parser.parse_args()

    if args.step == "check":
        compile_check_all()
        return

    if args.step in ("fleurs", "all"):
        submit_and_wait_wave(build_fleurs_wave_specs(), max_wait_s=7200, sequential=True)

    if args.step in ("longform", "all"):
        submit_and_wait_wave(build_longform_wave_specs(), max_wait_s=9000, sequential=True)

    if args.step in ("fr-wave3a", "all"):
        submit_and_wait_wave([build_fr_pt_spec(), build_fr_ct2_spec()], max_wait_s=10800, sequential=True)

    if args.step in ("fr-dual", "all"):
        specs = build_fr_dual_specs()
        for attempt in range(1, 3):
            submit_and_wait_wave(specs, max_wait_s=3600, queued_timeout_s=180, sequential=False)
            import json
            d0 = json.loads((specs[0].out_dir / "result.json").read_text(encoding="utf-8"))
            d1 = json.loads((specs[1].out_dir / "result.json").read_text(encoding="utf-8"))
            w0, w1 = d0["utc_inference_window"], d1["utc_inference_window"]
            overlap = max(0.0, min(w0[1], w1[1]) - max(w0[0], w1[0]))
            print(f"[fr-dual attempt {attempt}] utc_overlap={overlap:.2f}s (w0={w0[1]-w0[0]:.1f}s, w1={w1[1]-w1[0]:.1f}s)")
            if overlap > 0.0:
                break
            if attempt == 2:
                raise RuntimeError("Dual notebooks completed sequentially without UTC inference overlap on both attempts.")

    if args.step in ("summarize", "all"):
        write_full_summary()

    if args.step in ("cleanup", "all"):
        cleanup_ledger()
        out, _ = run_kaggle(["kernels", "list", "--mine", "--page-size", "50"], check=False)
        leftovers = [line for line in out.splitlines() if "chep-eval-" in line]
        print("Remaining campaign kernels:", leftovers)


if __name__ == "__main__":
    main()
