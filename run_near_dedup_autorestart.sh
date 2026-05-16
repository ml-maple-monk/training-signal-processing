#!/usr/bin/env bash
# Auto-restart wrapper for near_dedup Phase 1.
# Restarts on crash (e.g. PostgreSQL connection drop) with backoff.
# Incremental mode means restarts are safe — picks up from watermark.

CONN="postgresql://corpus:corpus_secret@localhost:5432/corpus"
WORKERS=8
LOG="/home/geeyang/workspace/training-signal-processing/near_dedup.log"
MAX_RESTARTS=50
RETRY_DELAY=30   # seconds between restarts

cd /home/geeyang/workspace/training-signal-processing

restarts=0
while [ $restarts -lt $MAX_RESTARTS ]; do
    echo "$(date '+%Y-%m-%d %H:%M:%S') INFO [autorestart] attempt $((restarts+1))/$MAX_RESTARTS" >> "$LOG"
    python -m training_signal_processing.pipelines.near_dedup.run \
        --conn "$CONN" \
        --workers "$WORKERS" >> "$LOG" 2>&1
    exit_code=$?

    # Exit 0 = completed normally, no restart needed
    if [ $exit_code -eq 0 ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') INFO [autorestart] completed successfully" >> "$LOG"
        exit 0
    fi

    restarts=$((restarts+1))
    echo "$(date '+%Y-%m-%d %H:%M:%S') WARN [autorestart] exited with code $exit_code, restarting in ${RETRY_DELAY}s ($restarts/$MAX_RESTARTS)" >> "$LOG"
    sleep $RETRY_DELAY
done

echo "$(date '+%Y-%m-%d %H:%M:%S') ERROR [autorestart] max restarts ($MAX_RESTARTS) reached, giving up" >> "$LOG"
exit 1
