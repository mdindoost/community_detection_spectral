#!/bin/bash
# Run LFR benchmark experiments

cd "$(dirname "$0")"
source venv/bin/activate

# Clean up previous LFR results
echo "Cleaning up previous LFR results..."
rm -rf results/lfr/*.json
rm -rf results/lfr/*.txt
rm -rf temp_lfr/*

# Create directories
mkdir -p results/lfr
mkdir -p logs

echo ""
echo "=========================================="
echo "LFR BENCHMARK EXPERIMENT"
echo "=========================================="
echo ""

# Run small graphs (n=1000) for quick experiments
echo "Running LFR experiment with n=1000 nodes..."
echo "This tests all (μ, k_avg) combinations with 5 repetitions each."
echo ""
echo "Monitor progress with: tail -f logs/lfr_n1000.log"
echo ""

PYTHONUNBUFFERED=1 python experiments/lfr_experiment.py --n 1000 --repeats 5 2>&1 | tee logs/lfr_n1000.log

echo ""
echo "=========================================="
echo "n=1000 experiments complete!"
echo "=========================================="
echo ""

# Optionally run larger graphs (n=5000) for validation
read -p "Run validation with n=5000 nodes? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Running LFR experiment with n=5000 nodes..."
    echo "This will take longer but provides more reliable results."
    echo ""

    PYTHONUNBUFFERED=1 python experiments/lfr_experiment.py --n 5000 --repeats 3 2>&1 | tee logs/lfr_n5000.log

    echo ""
    echo "=========================================="
    echo "n=5000 experiments complete!"
    echo "=========================================="
fi

echo ""
echo "Results saved to results/lfr/"
echo "  - lfr_n1000_results.json  (raw data)"
echo "  - lfr_n1000_summary.json  (aggregated)"
echo "  - lfr_n1000_report.txt    (human-readable)"
