#!/usr/bin/env bash
#
# Run one Part I campaign end to end. One invocation per campaign, which is one screen
# session per campaign on a shared machine.
#
#   ./scripts/songdo_static.sh budget              # first: the operating point
#   ./scripts/songdo_static.sh tuning              # at that operating point
#   ./scripts/songdo_static.sh oracle_tuning       # the Part II reference estimator
#   ./scripts/songdo_static.sh threshold_scaling   # only if tuning did not run at the final ratio
#   ./scripts/songdo_static.sh detectors
#   ./scripts/songdo_static.sh detectors_default   # default thresholds, evaluation scenes
#   ./scripts/songdo_static.sh detectors_hover     # hover-scale perturbation
#   ./scripts/songdo_static.sh estimators
#   ./scripts/songdo_static.sh rotation
#
# Detached, with the log surviving the session:
#
#   screen -S detectors -dm bash -c 'source .venv/bin/activate && ./scripts/songdo_static.sh detectors'
#   screen -r detectors          # reattach        Ctrl-A D   detach        screen -ls   list
#
# The campaign name selects the scene list (tuning or evaluation scenes).
#
# Run from the repository root. Scenes are read from $SCENES (default: data/songdo/static, where
# `fetch` extracts the record) and results written to $RESULTS_DIR (default: $SCENES/results,
# beside the published ones; set it to rerun into a directory of your own and compare).

set -euo pipefail

CAMPAIGN="${1:-}"
CONFIG_DIR="configs/songdo/static"
SCENES="${SCENES:-data/songdo/static}"
RESULTS_DIR="${RESULTS_DIR:-$SCENES/results}"
LOG_DIR="${LOG_DIR:-data/songdo/logs/static}"
ROTATION_DEGREES=(0 5 10 15 20 30 45 60 90)

usage() {
    echo "usage: $0 {budget|tuning|oracle_tuning|threshold_scaling|detectors|detectors_default|detectors_hover|estimators|rotation} [extra stabilo-benchmark flags]" >&2
    echo "       RESULTS_DIR=<dir> $0 tuning     (rerun into <dir>/tuning)" >&2
    exit 2
}

case "$CAMPAIGN" in
    tuning)     CONFIG=$CONFIG_DIR/tuning.yaml;         LIST="$CONFIG_DIR/scenes_tuning.txt" ;;
    oracle_tuning)
                CONFIG=$CONFIG_DIR/oracle_tuning.yaml;  LIST="$CONFIG_DIR/scenes_tuning.txt" ;;
    threshold_scaling)
                CONFIG=$CONFIG_DIR/threshold_scaling.yaml; LIST="$CONFIG_DIR/scenes_tuning.txt" ;;
    detectors)  CONFIG=$CONFIG_DIR/detectors.yaml;      LIST="$CONFIG_DIR/scenes_eval.txt" ;;
    detectors_default)
                CONFIG=$CONFIG_DIR/detectors_default.yaml; LIST="$CONFIG_DIR/scenes_eval.txt" ;;
    detectors_hover)
                CONFIG=$CONFIG_DIR/detectors_hover.yaml;   LIST="$CONFIG_DIR/scenes_eval.txt" ;;
    estimators) CONFIG=$CONFIG_DIR/estimators.yaml;     LIST="$CONFIG_DIR/scenes_eval.txt" ;;
    rotation)   CONFIG=$CONFIG_DIR/rotation_sweep.yaml; LIST="$CONFIG_DIR/scenes_eval.txt" ;;
    budget)     CONFIG=$CONFIG_DIR/budget.yaml;        LIST="$CONFIG_DIR/scenes_tuning.txt" ;;
    *)          usage ;;
esac
shift

# ---- Preflight ---------------------------------------------------------------------------
# Paths and annotations; `static` itself checks OpenCV and the detectors.

[[ -f "$CONFIG" ]] || { echo "Run this from the repository root: $CONFIG not found." >&2; exit 1; }

command -v stabilo-benchmark >/dev/null || { echo "stabilo-benchmark is not on PATH: activate the environment first (source .venv/bin/activate)." >&2; exit 1; }

[[ -d "$SCENES" ]] || { echo "No scenes at $SCENES: fetch the record (see the README) or set SCENES." >&2; exit 1; }

for subdirectory in images labels_hbb labels_obb; do
    [[ -d "$SCENES/$subdirectory" ]] || { echo "Not annotated: $SCENES/$subdirectory is missing. See step 2 of the README." >&2; exit 1; }
done

if [[ -d "$SCENES/labels_confidence" ]]; then
    echo -n "HBB fallback share: "
    cat "$SCENES"/labels_confidence/*.txt | awk '$1 == 0 {n++} END {print n+0, "of", NR, "boxes"}'
fi

mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/$CAMPAIGN.log"

echo "campaign : $CAMPAIGN"
echo "config   : $CONFIG"
echo "scenes   : $SCENES  ($(grep -cvE '^\s*(#|$)' "$LIST") listed in $(basename "$LIST"))"
echo "results  : $RESULTS_DIR"
echo "log      : $LOG"
echo

# ---- Run ---------------------------------------------------------------------------------
# The rotation sweep is one invocation, and one result directory, per magnitude.

run_one() {
    stabilo-benchmark static "$CONFIG" \
        --scenes "$SCENES" --scenes-list "$LIST" --results-dir "$RESULTS_DIR" -v 1 "$@"
}

{
    date -u +'started  : %Y-%m-%d %H:%M:%S UTC'
    if [[ "$CAMPAIGN" == rotation ]]; then
        for degrees in "${ROTATION_DEGREES[@]}"; do
            echo "--- rotation ${degrees} deg ---"
            run_one --set perturbation.geometric.rotation_deg="[$degrees,$degrees]" \
                    --name "rotation_${degrees}deg" "$@"
        done
    else
        run_one "$@"
    fi
    date -u +'finished : %Y-%m-%d %H:%M:%S UTC'
} 2>&1 | tee -a "$LOG"

# ---- What to do next ---------------------------------------------------------------------

echo
if [[ "$CAMPAIGN" == tuning ]]; then
    echo "Next: read the winning (filter_ratio, ransac_epipolar_threshold) per preset off"
    echo
    echo "    stabilo-benchmark report $RESULTS_DIR/tuning --kind static"
    echo "    # -> reports/tables/tuned_parameters.tex"
    echo
    echo "and write each into its preset in stabilo_benchmark/cfg/pipelines.yaml before any"
    echo "evaluation campaign runs. Select on maa, never on an HEA at one threshold."
elif [[ "$CAMPAIGN" == threshold_scaling ]]; then
    echo "Next: stabilo-benchmark report $RESULTS_DIR/threshold_scaling --kind static"
    echo "Per preset, compare the maa-optimal ransac_epipolar_threshold at downsample_ratio 0.18"
    echo "with the one at 0.50 (see configs/songdo/static/threshold_scaling.yaml)."
elif [[ "$CAMPAIGN" == budget ]]; then
    echo "Next: stabilo-benchmark report $RESULTS_DIR/budget --kind static"
    echo "Read maa against time_median per preset under both maskings, pick the operating point"
    echo "(see configs/songdo/static/budget.yaml), write it into cfg/pipelines.yaml, then run tuning."
else
    echo "Next: stabilo-benchmark report $RESULTS_DIR/$CAMPAIGN --kind static"
fi
echo "A campaign that stopped part way resumes with -r, which validates the stored grid first."
echo "A grid changed in the config runs only its new points with --regrid; stored runs still in it are kept."
