#!/usr/bin/env bash
#
# Run one Part II campaign over every case of its list, then the pooled report. One invocation
# per campaign, one screen session per campaign on a shared machine. Campaigns are named
# <regime>_<campaign>: offline (full resolution, Part I's feature budget) or realtime (the
# cheapest budget of Part I's budget campaign).
#
#   ./scripts/songdo_dynamic.sh offline_tuning     # tuning cases: per-pipeline r / theta
#   ./scripts/songdo_dynamic.sh offline_masking    # evaluation cases, then audit --results
#   ./scripts/songdo_dynamic.sh offline_margin     # evaluation cases: mask margin sweep
#   ./scripts/songdo_dynamic.sh offline_repeats    # learned pipelines, 3 repeats, one case
#   ./scripts/songdo_dynamic.sh realtime_tuning    # tuning cases at the real-time budget
#   ./scripts/songdo_dynamic.sh realtime_masking   # evaluation cases, then audit --results
#
# Detached, with the log surviving the session:
#
#   screen -S offline_masking -dm bash -c 'source .venv/bin/activate && ./scripts/songdo_dynamic.sh offline_masking'
#
# Run from the repository root. Cases are read from $DATA (default: data/songdo, where `fetch`
# extracts the record) and each case's results written to $RESULTS_DIR/<case> (default: inside the
# case, beside the published ones; set it to rerun into a directory of your own and compare).
# Before anything runs, every case must be prepared and annotated and carry
# the exclusion polygons its config names, its cached reference estimates must come from the
# oracle the config names, and, for the evaluation campaigns, its reference estimator must already
# be audited: `audit` precedes `dynamic`.
# Extra flags pass through to `stabilo-benchmark dynamic`; -r resumes.

set -euo pipefail

CAMPAIGN="${1:-}"
CONFIG_DIR="configs/songdo/dynamic"
DATA="${DATA:-data/songdo}"
RESULTS_DIR="${RESULTS_DIR:-}"
LOG_DIR="${LOG_DIR:-data/songdo/logs/dynamic}"
LEARNED="[xfeat,disk_lightglue,dedode_lightglue,keynet_lightglue,loftr]"

usage() {
    echo "usage: $0 {offline_tuning|offline_masking|offline_margin|offline_repeats|realtime_tuning|realtime_masking}" >&2
    echo "          [extra stabilo-benchmark dynamic flags]" >&2
    echo "       CASE=<case name> $0 offline_repeats      (default: the first case of cases.txt)" >&2
    echo "       RESULTS_DIR=<dir> $0 offline_masking     (rerun into <dir>/<case>/offline_masking)" >&2
    exit 2
}

EXTRA_SET=()
# The case drawn in the GCP audit figure of the masking reports.
AUDIT_CASE=2022-10-06_D9_AM3_S2

case "$CAMPAIGN" in
    offline_tuning|realtime_tuning)
        CONFIG="$CONFIG_DIR/$CAMPAIGN.yaml"; LIST="$CONFIG_DIR/cases_tuning.txt"; ROOT="$DATA/dynamic_tuning"; AUDITED=0 ;;
    offline_masking|offline_margin|realtime_masking)
        CONFIG="$CONFIG_DIR/$CAMPAIGN.yaml"; LIST="$CONFIG_DIR/cases.txt"; ROOT="$DATA/dynamic"; AUDITED=1 ;;
    offline_repeats)
        CONFIG="$CONFIG_DIR/offline_masking.yaml"; LIST="$CONFIG_DIR/cases.txt"; ROOT="$DATA/dynamic"; AUDITED=1
        EXTRA_SET=(--set repeats=3 --set "pipeline.preset=$LEARNED" --name offline_repeats) ;;
    *)  usage ;;
esac
shift

# ---- Preflight ---------------------------------------------------------------------------

[[ -f "$CONFIG" ]] || { echo "Run this from the repository root: $CONFIG not found." >&2; exit 1; }
command -v stabilo-benchmark >/dev/null || { echo "stabilo-benchmark is not on PATH: activate the environment first (source .venv/bin/activate)." >&2; exit 1; }
[[ -f "$LIST" ]] || { echo "No case list at $LIST." >&2; exit 1; }

CASES=()
while read -r name _; do
    [[ -n "$name" && "$name" != \#* ]] && CASES+=("$name")
done < "$LIST"
if [[ "$CAMPAIGN" == offline_repeats ]]; then
    CASES=("${CASE:-${CASES[0]}}")
fi
[[ ${#CASES[@]} -gt 0 ]] || { echo "$LIST lists no cases." >&2; exit 1; }

EXCLUSION=$(python -c "
import sys
from stabilo_benchmark.config import load_config
print((load_config(sys.argv[1]).get('metrics', {}).get('region', {}) or {}).get('exclusion') or '')
" "$CONFIG")

problems=0
for name in "${CASES[@]}"; do
    dir="$ROOT/$name"
    for required in CASE.json images labels_hbb labels_obb oracle/ORACLE.json; do
        [[ -e "$dir/$required" ]] || { echo "$name: missing $required" >&2; problems=1; }
    done
    if [[ -f "$dir/oracle/ORACLE.json" ]]; then
        python -c "
import sys
from stabilo_benchmark.config import load_config
from stabilo_benchmark.runners.dynamic import cached_oracle_matches
sys.exit(0 if cached_oracle_matches(sys.argv[1], load_config(sys.argv[2])) else 1)
" "$dir" "$CONFIG" || { echo "$name: cached reference estimates come from another oracle; run: stabilo-benchmark oracle $dir -c $CONFIG --overwrite" >&2; problems=1; }
    fi
    if [[ -n "$EXCLUSION" && ! -f "$dir/$EXCLUSION" ]]; then
        echo "$name: missing $EXCLUSION (polygons over above-ground structure on the reference frame)" >&2
        problems=1
    fi
    if [[ "$AUDITED" == 1 && ! -f "$dir/oracle/AUDIT.json" ]]; then
        echo "$name: not audited; annotate its ground control points and run: stabilo-benchmark audit $dir" >&2
        problems=1
    fi
done
[[ "$problems" == 0 ]] || { echo "Preflight failed; nothing was run." >&2; exit 1; }

echo -n "HBB fallback share: "
for name in "${CASES[@]}"; do cat "$ROOT/$name"/labels_confidence/*.txt; done | awk '$1 == 0 {n++} END {print n+0, "of", NR, "boxes"}'

mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/$CAMPAIGN.log"
RESULT_NAME="$CAMPAIGN"

# Where a case's campaign directory is: inside the case, or under $RESULTS_DIR/<case>.
results_of() { if [[ -n "$RESULTS_DIR" ]]; then echo "$RESULTS_DIR/$1"; else echo "$ROOT/$1/results"; fi; }

echo "campaign : $CAMPAIGN"
echo "config   : $CONFIG"
echo "cases    : ${#CASES[@]} from $(basename "$LIST"), under $ROOT"
echo "results  : $(results_of '<case>')/$RESULT_NAME"
echo "log      : $LOG"
echo

# ---- Run ---------------------------------------------------------------------------------

{
    date -u +'started  : %Y-%m-%d %H:%M:%S UTC'
    for name in "${CASES[@]}"; do
        echo "--- $name ---"
        stabilo-benchmark dynamic "$ROOT/$name" -c "$CONFIG" --results-dir "$(results_of "$name")" -v 1 \
            ${EXTRA_SET[@]+"${EXTRA_SET[@]}"} "$@"
        if [[ "$CAMPAIGN" == *_masking ]]; then
            stabilo-benchmark audit "$ROOT/$name" --results "$(results_of "$name")/$RESULT_NAME"
        fi
    done
    date -u +'finished : %Y-%m-%d %H:%M:%S UTC'
} 2>&1 | tee -a "$LOG"

# ---- What to do next ---------------------------------------------------------------------

DIRS=()
for name in "${CASES[@]}"; do DIRS+=("$(results_of "$name")/$RESULT_NAME"); done

echo
case "$CAMPAIGN" in
    offline_tuning|realtime_tuning)
        REGIME="${CAMPAIGN%_tuning}"
        echo "Next: stabilo-benchmark report ${DIRS[*]}"
        echo "then paste the two blocks of reports/tables/${CAMPAIGN}_tuned_parameters.yaml into the"
        echo "pipeline section of configs/songdo/dynamic/${REGIME}_masking.yaml$([[ $REGIME == offline ]] && echo ' and offline_margin.yaml')"
        echo "(check the report's warnings for values at an end of the swept range first)." ;;
    offline_repeats)
        echo "Next: stabilo-benchmark report ${DIRS[*]} --kind dynamic"
        echo "(writes tables/offline_repeats_repeat_spread.tex: the run-to-run spread against the audited floor)" ;;
    *_masking)
        echo "Next: stabilo-benchmark report ${DIRS[*]} --audit-case $AUDIT_CASE" ;;
    *)
        echo "Next: stabilo-benchmark report ${DIRS[*]}" ;;
esac
echo "A campaign that stopped part way resumes with -r, which validates the stored grid first."
echo "A grid changed in the config runs only its new points with --regrid; stored runs still in it are kept."
