#!/bin/bash
# Exp AB -- dataset-streaming driver.  Runs on the LOCAL machine; each network is
# rsynced to Fuji one at a time, run there detached, its results pulled back, and the
# dataset deleted before the next network starts.  Fuji has ~5.4 GB free, so we never
# hold two datasets at once and abort if free space drops below 1.5 GB.
set -u

REMOTE=mayooran@100.88.245.65
RREPO=md724/community_detection_spectral
LREPO=/home/md724/community_detection_spectral
EXP=PAPER_RESTRUCTURE/exp_AB_sparsifier_coverage
PY="\$HOME/md724/venv/bin/python"

NETS="ca-HepTh email-Eu-core ca-CondMat wiki-Vote email-Enron com-Amazon com-DBLP"
LABELLED="email-Eu-core com-Amazon com-DBLP"
METIS_NETS="email-Eu-core wiki-Vote"

cd "$LREPO" || exit 1
say() { echo "[$(date '+%F %T')] $*"; }

say "=== Exp AB streaming driver start ==="
ssh -o BatchMode=yes "$REMOTE" "mkdir -p $RREPO/$EXP $RREPO/datasets"
rsync -az "$LREPO/$EXP/run.py" "$REMOTE:$RREPO/$EXP/"
ssh -o BatchMode=yes "$REMOTE" "cd $RREPO/$EXP && rm -f results.csv recovery.csv skipped.csv"

for NET in $NETS; do
    FREE=$(ssh -o BatchMode=yes "$REMOTE" "df -BM --output=avail /home | tail -1 | tr -dc '0-9'")
    say "$NET: Fuji free = ${FREE} MB"
    if [ -z "$FREE" ] || [ "$FREE" -lt 1500 ]; then
        say "ABORT: less than 1500 MB free on Fuji before $NET"
        exit 2
    fi

    say "$NET: rsync dataset"
    rsync -az --relative "datasets/./$NET/$NET.txt" "$REMOTE:$RREPO/datasets/" || exit 3
    case " $LABELLED " in
        *" $NET "*) rsync -az --relative "datasets/./$NET/${NET}_labels.txt" \
                        "$REMOTE:$RREPO/datasets/" || exit 3 ;;
    esac

    FLAG=""
    case " $METIS_NETS " in *" $NET "*) FLAG="--metis" ;; esac

    say "$NET: launching detached run $FLAG"
    # completion is signalled by a sentinel file: pgrep -f would self-match both the
    # launcher shell and the polling shell, so it can never report DONE.
    ssh -o BatchMode=yes "$REMOTE" \
        "cd $RREPO/$EXP && rm -f $NET.done && setsid nohup sh -c '$PY run.py net $NET $FLAG > $NET.log 2>&1; touch $NET.done' > /dev/null 2>&1 < /dev/null & echo launched"

    sleep 20
    while true; do
        ST=$(ssh -o BatchMode=yes "$REMOTE" "test -f $RREPO/$EXP/$NET.done && echo DONE || echo RUNNING")
        [ "$ST" = "DONE" ] && break
        sleep 60
    done
    say "$NET: run finished; pulling results"

    scp -q "$REMOTE:$RREPO/$EXP/results.csv"  "$LREPO/$EXP/" 2>/dev/null
    scp -q "$REMOTE:$RREPO/$EXP/recovery.csv" "$LREPO/$EXP/" 2>/dev/null
    scp -q "$REMOTE:$RREPO/$EXP/skipped.csv"  "$LREPO/$EXP/" 2>/dev/null
    scp -q "$REMOTE:$RREPO/$EXP/$NET.log"     "$LREPO/$EXP/" 2>/dev/null

    say "$NET: deleting dataset from Fuji"
    ssh -o BatchMode=yes "$REMOTE" "rm -rf $RREPO/datasets/$NET"

    say "$NET: commit + push"
    git add "$EXP"/*.csv "$EXP"/*.log "$EXP"/run.py "$EXP"/stream_driver.sh 2>/dev/null
    git commit -q -m "Exp AB: $NET complete (7 sparsifier arms, honest transfer, fragments)" \
        && git push -q origin refactor_v2 && say "$NET: pushed"
done

say "=== Exp AB streaming driver done ==="
