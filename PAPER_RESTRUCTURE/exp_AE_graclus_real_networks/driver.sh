#!/bin/bash
# exp_AE: Graclus mirror of exp_AD, same grid, same order (small networks first).
cd "$(dirname "$0")"
PY=~/md724/venv/bin/python
run () { echo "=== $1 --k-mode $2 ==="; timeout 21600 $PY run.py net "$1" --graclus-only --k-mode "$2" 2>&1 | tail -20; }
run email-Eu-core gt
run wiki-Vote     nc_base
run ca-HepTh      nc_base
run ca-CondMat    nc_base
run email-Enron   nc_base
run com-Amazon    nc_base
run com-Amazon    gt
run com-DBLP      nc_base
run com-DBLP      gt
echo "ALL DONE"
