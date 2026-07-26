#!/bin/bash
# Exp AC driver -- one sequential capped job (EXPLORATION.md rule 7).
# Registered generators (sbm, dcsbm) run FIRST; the added 2x2 controls after.
set -u
cd "$(dirname "$0")" || exit 1
PY="${PY:-$HOME/md724/venv/bin/python}"
export AC_CACHE="${AC_CACHE:-$HOME/md724/ac_cache}"
mkdir -p "$AC_CACHE"

echo "=== $(date -Is) gen ==="
$PY run.py gen > gen.log 2>&1
echo "gen rc=$?"

for G in sbm dcsbm sbm_ps dcsbm_ps; do
  echo "=== $(date -Is) armA $G ==="
  $PY run.py armA "$G" > "armA_${G}.log" 2>&1
  echo "armA $G rc=$?"
  echo "=== $(date -Is) armB $G ==="
  $PY run.py armB "$G" > "armB_${G}.log" 2>&1
  echo "armB $G rc=$?"
done

for G in sbm dcsbm sbm_ps dcsbm_ps; do
  echo "=== $(date -Is) metisseeds $G ==="
  $PY run.py metisseeds "$G" > "metisseeds_${G}.log" 2>&1
  echo "metisseeds $G rc=$?"
done

echo "=== $(date -Is) ALL DONE ==="
touch driver.done
