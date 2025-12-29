#!/bin/bash
# Run experiments on classic benchmark graphs

cd "$(dirname "$0")"
source venv/bin/activate

# Create directories
mkdir -p logs

echo "=========================================="
echo "CLASSIC BENCHMARK GRAPHS EXPERIMENT"
echo "=========================================="
echo ""

# Classic benchmarks with ground truth (run first)
BENCHMARKS_GT=("karate" "dolphins" "football" "polbooks" "polblogs")

# Classic benchmarks without ground truth
BENCHMARKS_NO_GT=("lesmis" "jazz" "celegans" "netscience" "power")

echo "Running benchmarks WITH ground truth..."
for dataset in "${BENCHMARKS_GT[@]}"; do
    echo "Starting: $dataset"
    PYTHONUNBUFFERED=1 python experiments/community_experiment.py --datasets "$dataset" > "logs/${dataset}.log" 2>&1 &
done
wait
echo "Ground truth benchmarks complete!"
echo ""

echo "Running benchmarks WITHOUT ground truth..."
for dataset in "${BENCHMARKS_NO_GT[@]}"; do
    echo "Starting: $dataset"
    PYTHONUNBUFFERED=1 python experiments/community_experiment.py --datasets "$dataset" > "logs/${dataset}.log" 2>&1 &
done
wait
echo "All benchmarks complete!"
echo ""

echo "=========================================="
echo "Results Summary"
echo "=========================================="

# Regenerate report
python gather_results.py

echo ""
echo "Datasets with ground truth (NMI/ARI available):"
for dataset in "${BENCHMARKS_GT[@]}"; do
    if [ -f "results/${dataset}/results.json" ]; then
        echo "  + $dataset"
    else
        echo "  - $dataset (FAILED - check logs/${dataset}.log)"
    fi
done

echo ""
echo "Datasets without ground truth (modularity only):"
for dataset in "${BENCHMARKS_NO_GT[@]}"; do
    if [ -f "results/${dataset}/results.json" ]; then
        echo "  + $dataset"
    else
        echo "  - $dataset (FAILED - check logs/${dataset}.log)"
    fi
done
