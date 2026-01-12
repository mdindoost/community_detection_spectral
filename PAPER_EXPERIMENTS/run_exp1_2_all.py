#!/usr/bin/env python3
"""
Driver script for Experiment 1.2: Theoretical Predictions Validation

Runs exp1_2_theoretical_predictions.py on all specified datasets and
collects results in a consistent directory structure.

Usage:
    python run_exp1_2_all.py [--datasets DATASET1,DATASET2,...] [--parallel N]

Output:
    results/exp1_2_theoretical/<dataset>/
        - <dataset>_theoretical_validation_FIXED.csv
        - <dataset>_summary_FIXED.csv
        - <dataset>_ratio_validation.png
        - <dataset>_modularity_change.png
"""

import subprocess
import sys
from pathlib import Path
import argparse
from datetime import datetime
import json

# Default datasets for the experiment
DEFAULT_DATASETS = [
    "ca-AstroPh",
    "ca-CondMat",
    "ca-GrQc",
    "ca-HepPh",
    "ca-HepTh",
    "cit-HepPh",
    "cit-HepTh",
    "email-Enron",
    "facebook-combined",
    "ego-Facebook",
    "wiki-Vote",
    "email-Eu-core",
]

# Script location
SCRIPT_DIR = Path(__file__).parent
EXPERIMENT_SCRIPT = SCRIPT_DIR / "exp1_2_theoretical_predictions.py"
RESULTS_BASE = SCRIPT_DIR / "results" / "exp1_2_theoretical"


def run_dataset(dataset: str, verbose: bool = True) -> dict:
    """
    Run experiment on a single dataset.

    Returns:
        dict with status, runtime, and output paths
    """
    start_time = datetime.now()

    if verbose:
        print(f"\n{'='*80}")
        print(f"Running Experiment 1.2 on: {dataset}")
        print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")

    # Run the experiment
    cmd = [sys.executable, str(EXPERIMENT_SCRIPT), dataset]

    try:
        result = subprocess.run(
            cmd,
            cwd=str(SCRIPT_DIR),
            capture_output=not verbose,
            text=True,
            timeout=3600  # 1 hour timeout per dataset
        )

        end_time = datetime.now()
        runtime = (end_time - start_time).total_seconds()

        # Check output files
        output_dir = RESULTS_BASE
        summary_file = output_dir / f"{dataset}_summary_FIXED.csv"
        raw_file = output_dir / f"{dataset}_theoretical_validation_FIXED.csv"

        status = {
            "dataset": dataset,
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "runtime_seconds": runtime,
            "summary_exists": summary_file.exists(),
            "raw_exists": raw_file.exists(),
            "summary_path": str(summary_file) if summary_file.exists() else None,
            "raw_path": str(raw_file) if raw_file.exists() else None,
        }

        if verbose:
            if status["success"]:
                print(f"\n[SUCCESS] {dataset} completed in {runtime:.1f}s")
            else:
                print(f"\n[FAILED] {dataset} (return code: {result.returncode})")
                if not verbose:
                    print(f"STDERR: {result.stderr[:500]}")

        return status

    except subprocess.TimeoutExpired:
        return {
            "dataset": dataset,
            "success": False,
            "error": "timeout",
            "runtime_seconds": 3600,
        }
    except Exception as e:
        return {
            "dataset": dataset,
            "success": False,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Run Experiment 1.2 on multiple datasets"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated list of datasets (default: all)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress experiment output"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print datasets without running"
    )

    args = parser.parse_args()

    # Determine datasets to run
    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",")]
    else:
        datasets = DEFAULT_DATASETS

    print("="*80)
    print("EXPERIMENT 1.2: THEORETICAL PREDICTIONS VALIDATION")
    print("="*80)
    print(f"\nDatasets to process ({len(datasets)}):")
    for i, d in enumerate(datasets, 1):
        print(f"  {i:2d}. {d}")

    if args.dry_run:
        print("\n[DRY RUN] Exiting without running experiments.")
        return

    print(f"\nResults will be saved to: {RESULTS_BASE}")
    print("\n" + "-"*80)

    # Run all datasets
    results = []
    successful = 0
    failed = 0

    start_all = datetime.now()

    for i, dataset in enumerate(datasets, 1):
        print(f"\n[{i}/{len(datasets)}] Processing {dataset}...")

        status = run_dataset(dataset, verbose=not args.quiet)
        results.append(status)

        if status.get("success"):
            successful += 1
        else:
            failed += 1

    end_all = datetime.now()
    total_runtime = (end_all - start_all).total_seconds()

    # Summary
    print("\n" + "="*80)
    print("EXPERIMENT 1.2 COMPLETE")
    print("="*80)
    print(f"\nTotal runtime: {total_runtime:.1f}s ({total_runtime/60:.1f} minutes)")
    print(f"Successful: {successful}/{len(datasets)}")
    print(f"Failed: {failed}/{len(datasets)}")

    if failed > 0:
        print("\nFailed datasets:")
        for r in results:
            if not r.get("success"):
                print(f"  - {r['dataset']}: {r.get('error', 'unknown error')}")

    # Save run log
    log_file = RESULTS_BASE / "run_log.json"
    log_data = {
        "timestamp": start_all.isoformat(),
        "total_runtime_seconds": total_runtime,
        "successful": successful,
        "failed": failed,
        "results": results,
    }

    RESULTS_BASE.mkdir(parents=True, exist_ok=True)
    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=2)

    print(f"\nRun log saved to: {log_file}")

    # List output files
    print("\nOutput files:")
    for r in results:
        if r.get("summary_path"):
            print(f"  {r['dataset']}: {r['summary_path']}")


if __name__ == "__main__":
    main()
