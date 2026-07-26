#!/bin/bash
# ca-CondMat was run before the operating-point rounding fix: its backbone operating
# point was rounded DOWN to 0.2340 while the MST floor is 0.234011, so both backbone
# arms were skipped at the one point that exists for them.  This waits for the main
# streaming driver to finish, then drops ca-CondMat's rows and re-runs it.
set -u
REMOTE=mayooran@100.88.245.65
RREPO=md724/community_detection_spectral
LREPO=/home/md724/community_detection_spectral
EXP=PAPER_RESTRUCTURE/exp_AB_sparsifier_coverage
PY="\$HOME/md724/venv/bin/python"
cd "$LREPO" || exit 1
say() { echo "[$(date '+%F %T')] $*"; }

while pgrep -f "stream_driver.sh" > /dev/null; do sleep 60; done
say "main driver finished; starting ca-CondMat re-run"

FREE=$(ssh -o BatchMode=yes "$REMOTE" "df -BM --output=avail /home | tail -1 | tr -dc '0-9'")
say "Fuji free = ${FREE} MB"
[ -z "$FREE" ] || [ "$FREE" -lt 1500 ] && { say "ABORT: low disk"; exit 2; }

rsync -az --relative "datasets/./ca-CondMat/ca-CondMat.txt" "$REMOTE:$RREPO/datasets/" || exit 3
rsync -az "$LREPO/$EXP/run.py" "$REMOTE:$RREPO/$EXP/"

ssh -o BatchMode=yes "$REMOTE" "cd $RREPO/$EXP && \
  for f in results.csv recovery.csv skipped.csv; do \
    [ -f \$f ] && grep -v '^ca-CondMat,' \$f > \$f.tmp && mv \$f.tmp \$f; done; \
  rm -f ca-CondMat.done && \
  setsid nohup sh -c '$PY run.py net ca-CondMat > ca-CondMat.log 2>&1; touch ca-CondMat.done' \
    > /dev/null 2>&1 < /dev/null & echo launched"

sleep 20
while true; do
    ST=$(ssh -o BatchMode=yes "$REMOTE" "test -f $RREPO/$EXP/ca-CondMat.done && echo DONE || echo RUNNING")
    [ "$ST" = "DONE" ] && break
    sleep 60
done
say "ca-CondMat re-run finished; pulling results"
for f in results.csv recovery.csv skipped.csv ca-CondMat.log; do
    scp -q "$REMOTE:$RREPO/$EXP/$f" "$LREPO/$EXP/" 2>/dev/null
done
ssh -o BatchMode=yes "$REMOTE" "rm -rf $RREPO/datasets/ca-CondMat"
git add "$EXP"/*.csv "$EXP"/*.log "$EXP"/*.py "$EXP"/*.sh 2>/dev/null
git commit -q -m "Exp AB: ca-CondMat re-run with corrected backbone operating point" \
    && git push -q origin refactor_v2 && say "pushed"
say "=== fixup done ==="
