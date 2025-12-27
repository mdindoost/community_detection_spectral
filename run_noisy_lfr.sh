#!/bin/bash
# Run Noisy LFR benchmark experiments

cd "$(dirname "$0")"
source venv/bin/activate

# Clean up previous results
echo "Cleaning up previous noisy LFR results..."
rm -rf results/noisy_lfr/*.json
rm -rf results/noisy_lfr/*.txt

# Create directories
mkdir -p results/noisy_lfr
mkdir -p logs

echo ""
echo "=========================================="
echo "NOISY LFR BENCHMARK EXPERIMENT"
echo "=========================================="
echo ""
echo "HYPOTHESIS:"
echo "  Clean LFR: Sparsification hurts (removes signal)"
echo "  Noisy LFR: Sparsification helps (removes noise)"
echo ""

# Run with n=1000 for quick experiments
echo "Running noisy LFR experiment with n=1000 nodes..."
echo "Testing noise levels: 0%, 10%, 20%, 30%, 50%"
echo ""
echo "Monitor progress with: tail -f logs/noisy_lfr_n1000.log"
echo ""

PYTHONUNBUFFERED=1 python experiments/noisy_lfr_experiment.py --n 1000 --repeats 5 2>&1 | tee logs/noisy_lfr_n1000.log

echo ""
echo "=========================================="
echo "n=1000 experiments complete!"
echo "=========================================="
echo ""

# Optionally run with larger graphs
read -p "Run validation with n=5000 nodes? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Running noisy LFR experiment with n=5000 nodes..."
    PYTHONUNBUFFERED=1 python experiments/noisy_lfr_experiment.py --n 5000 --repeats 3 2>&1 | tee logs/noisy_lfr_n5000.log
    echo ""
    echo "n=5000 experiments complete!"
fi

echo ""
echo "Results saved to results/noisy_lfr/"
echo "  - noisy_lfr_n1000_results.json  (raw data)"
echo "  - noisy_lfr_n1000_summary.json  (aggregated)"
echo "  - noisy_lfr_n1000_report.txt    (human-readable)"
