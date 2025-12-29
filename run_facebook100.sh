#!/bin/bash
# Run experiments on all Facebook100 university networks

cd "$(dirname "$0")"
source venv/bin/activate

# Create directories
mkdir -p logs

echo "=========================================="
echo "FACEBOOK100 EXPERIMENT (100 Universities)"
echo "=========================================="
echo ""

# All Facebook100 datasets (sorted alphabetically)
FACEBOOK100=(
    "Amherst41" "Bowdoin47" "Brandeis99" "Brown11" "Bucknell39" "Cal65" "Caltech36"
    "Carnegie49" "Colgate88" "Columbia2" "Cornell5" "Dartmouth6" "Duke14" "Emory27"
    "Georgetown15" "GWU54" "Hamilton46" "Harvard1" "Haverford76" "Howard90" "Indiana69"
    "JMU79" "JohnsHopkins55" "Lehigh96" "Maine59" "Mich67" "Michigan23" "Middlebury45"
    "MIT8" "MSU24" "UCSB37" "UNC28" "Northeastern19" "Northwestern25" "NotreDame57"
    "NYU9" "Oberlin44" "Penn94" "Pepperdine86" "Princeton12" "Reed98" "Rice31"
    "Rochester38" "Rutgers89" "Santa74" "Simmons81" "Smith60" "Stanford3" "Swarthmore42"
    "Syracuse56" "Temple83" "Tennessee95" "Texas80" "Tufts18" "Tulane29" "UC33" "UC61"
    "UC64" "UCLA26" "UCSC68" "UChicago30" "UConn91" "UF21" "UIllinois20" "UMass92"
    "UPenn7" "USC35" "USF51" "USFCA72" "UVA16" "Vanderbilt48" "Vermont70" "Villanova62"
    "Virginia63" "Wake73" "WashU32" "Wellesley22" "Wesleyan43" "William77" "Williams40"
    "Wisconsin87" "Yale4"
)

TOTAL=${#FACEBOOK100[@]}
echo "Total datasets: $TOTAL"
echo ""

# Number of parallel jobs (adjust based on available memory)
MAX_PARALLEL=4

count=0
for dataset in "${FACEBOOK100[@]}"; do
    echo "[$((count+1))/$TOTAL] Starting: $dataset"
    PYTHONUNBUFFERED=1 python experiments/community_experiment.py --datasets "$dataset" > "logs/${dataset}.log" 2>&1 &

    count=$((count + 1))

    # Wait if we've hit the max parallel jobs
    if [ $((count % MAX_PARALLEL)) -eq 0 ]; then
        echo "  Waiting for batch to complete..."
        wait
        echo "  Batch complete. Progress: $count/$TOTAL"
        echo ""
    fi
done

# Wait for any remaining jobs
wait

echo "=========================================="
echo "All Facebook100 experiments complete!"
echo "=========================================="

# Regenerate report
echo ""
echo "Generating combined report..."
python gather_results.py

# Count successes/failures
echo ""
echo "Results summary:"
success=0
failed=0
for dataset in "${FACEBOOK100[@]}"; do
    if [ -f "results/${dataset}/results.json" ]; then
        success=$((success + 1))
    else
        failed=$((failed + 1))
        echo "  FAILED: $dataset (check logs/${dataset}.log)"
    fi
done
echo ""
echo "Completed: $success/$TOTAL"
if [ $failed -gt 0 ]; then
    echo "Failed: $failed"
fi
