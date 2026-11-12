#!/usr/bin/env bash
#
# Draw the polygons over above-ground structure that a case excludes from its evaluation region,
# on the case's reference frame, with LabelMe (`uv tool install labelme`).
#
#   ./scripts/draw_exclusion.sh <case dir> [<case dir> ...]
#
# Each case opens its reference frame alone. Draw with Ctrl+N (click the vertices, click the first
# one to close; any label), close the window: LabelMe saves on every edit. The polygons land in
# <case>/EXCLUSION.json. A case closed with nothing drawn can be recorded as having nothing to
# exclude. Re-running a case reopens its polygons.

set -euo pipefail

command -v labelme >/dev/null || { echo "labelme is not on PATH: uv tool install labelme" >&2; exit 1; }
[[ $# -gt 0 ]] || { echo "usage: $0 <case dir> [<case dir> ...]" >&2; exit 2; }

staging=$(mktemp -d)
trap 'rm -rf "$staging"' EXIT

n=0
for case_dir in "$@"; do
    n=$((n + 1))
    case_dir=${case_dir%/}
    [[ -f "$case_dir/CASE.json" ]] || { echo "$case_dir: no CASE.json" >&2; exit 1; }
    stem=$(python3 -c "import json, sys; print(f\"{json.load(open(sys.argv[1]))['reference_index']:06d}\")" "$case_dir/CASE.json")
    image="$case_dir/images/$stem.jpg"
    [[ -f "$image" ]] || { echo "$case_dir: reference frame $image not found" >&2; exit 1; }

    # LabelMe lists every image in the folder it opens, so the reference frame is opened from a
    # folder of its own; it reads and writes <stem>.json beside it.
    work="$staging/$(basename "$case_dir")"
    mkdir -p "$work"
    cp "$image" "$work/$stem.jpg"
    if [[ -f "$case_dir/EXCLUSION.json" ]]; then
        # The record names images/<stem>.jpg relative to the case; LabelMe resolves it beside the
        # staged copy, so the copy names the frame there.
        python3 - "$case_dir/EXCLUSION.json" "$work/$stem.json" "$stem.jpg" <<'EOF'
import json, sys
source, target, image = sys.argv[1:]
payload = json.load(open(source))
payload['imagePath'] = image
with open(target, 'w') as f:
    json.dump(payload, f, indent=2)
EOF
    fi

    echo "[$n/$#] $(basename "$case_dir"): draw, then close the window"
    labelme "$work/$stem.jpg" --output "$work" --labels structure >/dev/null 2>&1 || true

    if [[ ! -f "$work/$stem.json" ]]; then
        read -r -p "  nothing drawn: record this case as having nothing to exclude? [y/N] " answer
        [[ "$answer" == [yY]* ]] || { echo "  skipped; re-run this case later"; continue; }
        printf '{"shapes": []}\n' > "$work/$stem.json"
    fi
    python3 - "$work/$stem.json" "$case_dir/EXCLUSION.json" "images/$stem.jpg" <<'EOF'
import json, sys
source, target, image = sys.argv[1:]
payload = json.load(open(source))
payload['imagePath'], payload['imageData'] = image, None
with open(target, 'w') as f:
    json.dump(payload, f, indent=2)
polygons = sum(s.get('shape_type') == 'polygon' for s in payload['shapes'])
print(f'  {polygons} polygon(s) -> {target}')
EOF
done
