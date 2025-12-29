#!/bin/bash
# Run experiments on Facebook and social network datasets

cd "$(dirname "$0")"
source venv/bin/activate

# Create directories
mkdir -p logs

echo "=========================================="
echo "SOCIAL NETWORKS EXPERIMENT"
echo "=========================================="
echo ""

# Define dataset groups
FACEBOOK100_DATASETS=("Rice31" "Texas80" "Penn94")
SMALL_SOCIAL=("facebook-combined" "email-Enron")
LARGE_SOCIAL=("soc-Pokec")

echo "Running Facebook100 Networks (small, quick)..."
for dataset in "${FACEBOOK100_DATASETS[@]}"; do
    echo "Starting: $dataset"
    PYTHONUNBUFFERED=1 python experiments/community_experiment.py --datasets "$dataset" > "logs/${dataset}.log" 2>&1 &
done
wait
echo "Facebook100 networks complete!"
echo ""

echo "Running Small Social Networks..."
for dataset in "${SMALL_SOCIAL[@]}"; do
    echo "Starting: $dataset"
    PYTHONUNBUFFERED=1 python experiments/community_experiment.py --datasets "$dataset" > "logs/${dataset}.log" 2>&1 &
done
wait
echo "Small social networks complete!"
echo ""

echo "Running Large Social Networks (this will take a while)..."
echo "Monitor progress with: tail -f logs/soc-Pokec.log"
echo ""

for dataset in "${LARGE_SOCIAL[@]}"; do
    # Run one at a time for large datasets to avoid memory issues
    echo "Starting: $dataset"
    PYTHONUNBUFFERED=1 python experiments/community_experiment.py --datasets "$dataset" 2>&1 | tee "logs/${dataset}.log"
    echo "$dataset complete!"
    echo ""
done

echo "=========================================="
echo "All social network experiments complete!"
echo "=========================================="

# Regenerate report
echo ""
echo "Generating combined report..."
python gather_results.py

echo ""
echo "Results:"
for dataset in "${FACEBOOK100_DATASETS[@]}" "${SMALL_SOCIAL[@]}" "${LARGE_SOCIAL[@]}"; do
    if [ -f "results/${dataset}/results.json" ]; then
        echo "  + $dataset - completed"
    else
        echo "  - $dataset - failed (check logs/${dataset}.log)"
    fi
done
