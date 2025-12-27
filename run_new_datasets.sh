#!/bin/bash
# Run experiments on new datasets (large SNAP + citation + PPI)

cd "$(dirname "$0")"
source venv/bin/activate

# Create directories
mkdir -p logs

echo "=========================================="
echo "NEW DATASETS EXPERIMENT"
echo "=========================================="
echo ""

# Define dataset groups
CITATION_DATASETS=("cora" "citeseer")
PPI_DATASETS=("yeast-ppi" "human-ppi")
LARGE_SNAP_DATASETS=("com-LiveJournal" "com-Orkut")
# com-Friendster is VERY large (~65M nodes) - run separately if needed

MAX_JOBS=2

echo "Running Citation Networks (small, quick)..."
for dataset in "${CITATION_DATASETS[@]}"; do
    echo "Starting: $dataset"
    PYTHONUNBUFFERED=1 python experiments/community_experiment.py --datasets "$dataset" > "logs/${dataset}.log" 2>&1 &
done
wait
echo "Citation networks complete!"
echo ""

echo "Running PPI Networks (small, quick)..."
for dataset in "${PPI_DATASETS[@]}"; do
    echo "Starting: $dataset"
    PYTHONUNBUFFERED=1 python experiments/community_experiment.py --datasets "$dataset" > "logs/${dataset}.log" 2>&1 &
done
wait
echo "PPI networks complete!"
echo ""

echo "Running Large SNAP Datasets (this will take a while)..."
echo "Monitor progress with: tail -f logs/com-LiveJournal.log"
echo ""

for dataset in "${LARGE_SNAP_DATASETS[@]}"; do
    # Run one at a time for large datasets to avoid memory issues
    echo "Starting: $dataset"
    PYTHONUNBUFFERED=1 python experiments/community_experiment.py --datasets "$dataset" 2>&1 | tee "logs/${dataset}.log"
    echo "$dataset complete!"
    echo ""
done

echo "=========================================="
echo "All new datasets complete!"
echo "=========================================="

# Regenerate report
echo ""
echo "Generating combined report..."
python gather_results.py

echo ""
echo "Results:"
for dataset in "${CITATION_DATASETS[@]}" "${PPI_DATASETS[@]}" "${LARGE_SNAP_DATASETS[@]}"; do
    if [ -f "results/${dataset}/results.json" ]; then
        echo "  ✓ $dataset - completed"
    else
        echo "  ✗ $dataset - failed (check logs/${dataset}.log)"
    fi
done
