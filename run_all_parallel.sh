#!/bin/bash
# Run all datasets with max 3 parallel jobs

cd "$(dirname "$0")"
source venv/bin/activate

# Clean up previous results and cached files
echo "Cleaning up previous results..."
rm -rf results/*/results.json
rm -rf results/summary.json
rm -rf results_report.txt
rm -rf logs/*.log

# Optionally clean cached sparsification files (uncomment to force re-sparsification)
# rm -rf datasets/*/edges_spectral_*.txt

DATASETS=(
    "email-Eu-core"
    "wiki-Vote"
    "ca-HepPh"
    "soc-Epinions1"
    "com-DBLP"
    "com-Amazon"
    "com-Youtube"
)

MAX_JOBS=3

# Create logs directory
mkdir -p logs

echo "Starting experiments for ${#DATASETS[@]} datasets (max $MAX_JOBS parallel)..."
echo "Logs will be saved to logs/<dataset>.log"
echo ""
echo "Monitor live progress with: tail -f logs/*.log"
echo ""

# Run datasets with limited parallelism
for dataset in "${DATASETS[@]}"; do
    # Wait if we have MAX_JOBS running
    while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
        sleep 5
    done

    echo "Starting: $dataset"
    PYTHONUNBUFFERED=1 python experiments/community_experiment.py --datasets "$dataset" > "logs/${dataset}.log" 2>&1 &
done

# Wait for all remaining jobs to complete
echo ""
echo "All experiments launched. Waiting for completion..."
wait

echo ""
echo "All experiments finished!"
echo ""

# Show summary of results
echo "Results:"
for dataset in "${DATASETS[@]}"; do
    if [ -f "results/${dataset}/results.json" ]; then
        echo "  ✓ $dataset - completed"
    else
        echo "  ✗ $dataset - failed (check logs/${dataset}.log)"
    fi
done

# Generate combined report
echo ""
echo "Generating combined report..."
python gather_results.py
