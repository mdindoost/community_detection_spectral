#!/bin/bash
# Exp AD driver -- one sequential capped job (EXPLORATION.md rule 7).
# Small networks first so early results arrive early.
set -u
cd "$(dirname "$0")" || exit 1
PY="${PY:-$HOME/md724/venv/bin/python}"

NETS="ca-HepTh email-Eu-core ca-CondMat wiki-Vote email-Enron com-Amazon com-DBLP"
LABELLED="email-Eu-core com-DBLP com-Amazon"

for NET in $NETS; do
  echo "=== $(date -Is) $NET k_mode=nc_base ==="
  $PY run.py net "$NET" --metis-only --k-mode nc_base > "${NET}_ncbase.log" 2>&1
  echo "  rc=$?"
  case " $LABELLED " in *" $NET "*)
    echo "=== $(date -Is) $NET k_mode=gt ==="
    $PY run.py net "$NET" --metis-only --k-mode gt > "${NET}_gt.log" 2>&1
    echo "  rc=$?" ;;
  esac
done

echo "=== $(date -Is) ALL DONE ==="
touch driver.done
