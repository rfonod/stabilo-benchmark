# Songdo dynamic campaign: choosing and running Part II's cases

Part II's cases are chosen from the recordings alone, before any case is benchmarked, to span the
conditions under which a moving foreground can capture the homography.

Every case is benchmarked in two regimes, which differ only in the compute budget each pipeline gets:

| Regime | Budget | Campaigns |
| --- | --- | --- |
| **offline** | downsample ratio 0.5, 2000 features: registration after the flight | `offline_tuning`, `offline_masking`, `offline_margin`, `offline_repeats` |
| **realtime** | downsample ratio 0.18, 1000 features, the cheapest budget of Part I's budget campaign: registration at video rate | `realtime_tuning`, `realtime_masking` |

Both budgets are points of Part I's budget grid (`configs/songdo/static/budget.yaml`). Both regimes
use the same cases, pairs, reference estimator and ground control points, and give the inlier
threshold in processed-image pixels (`ransac_threshold_space: processed`), so Part I's tuned values
keep their meaning as the frames shrink. Each campaign's config is
`configs/songdo/dynamic/<campaign>.yaml` and its results land in `<case>/results/<campaign>`, or under
`--results-dir`.

## Design

Every benchmark pair registers a later frame onto the case's reference frame, and a wrong
homography can be fitted only to objects present in both frames that moved together. The survey
therefore reads each candidate the way the benchmark will: as pairs (reference, reference + k),
one k per bin of the config's `survey_bins` (1, 2, 5, 11 and 22 frames for `offline_masking.yaml`:
the first second, while the objects in view at the reference are still shared with it and a group
moving together still moves as one). In each pair it removes the platform's motion (an affine fit
to the background) and looks for every reference object among the later frame's detections by
appearance. Each reference object ends up as one of:

| Fate | Meaning |
| --- | --- |
| **static** | found within a tenth of an object width of where it was |
| **displaced** | found farther away: foreground that could carry a homography with it |
| **coherent** | the largest displaced group that moved together: displacements agreeing with one member's within a tenth of an object width or a quarter of that member's displacement, whichever is larger |
| **gone** | not found: left the frame, occluded, or no longer recognizable |

A tenth of an object width is clear of the apparent motion of a stationary object (box placement
and platform-motion error) and less than a car covers between consecutive frames at urban speeds,
so a stream is displaced from the first offset on. The relative part of the coherence tolerance lets a stream
whose members drift apart as it moves stay one group, while opposing or crossing streams never
agree. Each fate is a fraction of the frame, averaged over the offsets with equal weight. The design
reads two of them:

| Factor | Measured by |
| --- | --- |
| **Moving foreground** | `displaced`: how much of the frame the moving, shared foreground covers |
| **Coherence** | `coherent / displaced`: how much of it moves as one structure |
| **Stationarity** (control) | `static / (static + displaced)`: how much of the found foreground stays put |

Coherence is defined only when at least four objects are displaced on average: one or two always
move "together". A hover without that much motion is `unclassified` and in no cell.

Background texture (corner density on the static background) and total coverage are recorded with
every candidate. Opposing or crossing streams have low coherence however aligned their boxes are,
which is a predicted protective factor, not missing data.

- A 2×2 on moving foreground × coherence at the pool medians, two cases per cell, from different
  sites, ranked by their distance from the split boundaries, the weaker axis deciding.
- One stationary control: much foreground, little of it moving (stationarity in the pool's top
  decile, static foreground at least the pool median). If masking helps there as much as in a
  flowing case, the effect is texture removal, not moving objects. Every case also records
  per-pair `persistence`, so stationary foreground is a stratum within cases too.

That makes nine cases. Thresholds are in object widths, so the survey does not depend on the ground
resolution. Nothing is tracked through the recording: the survey annotates six frames per segment.

A case is one hover, the recording of one drone over one intersection, already cut to its stable
part, registered onto one reference frame. Traffic at an intersection comes in phases, so the
survey reads every hover at several points, one every 30 s clear of its first and last seconds
(`--anchor scan`), and a selected point becomes the case's reference. Candidates are the hovers of
one recording day at the evaluation sites of the static split
(`configs/songdo/static/scenes_eval.txt`, since the tuned thresholds are also used in Part II), at
points that leave the full 3600 frames after them, one case per hover at most. If a cell stays
`SHORT`, the hovers of the next day in a fixed order are added; no hover is picked by hand. The
pool is fixed before the survey runs:

| Pool | Value |
| --- | --- |
| Day | 2022-10-06, then 2022-10-05, 2022-10-04, 2022-10-07 |
| Sites | B, C, F, G, I, J, L, M, O, P, R, S, U |
| Hovers | every cut hover of those sites on that day, all ten sessions |

The hovers go in `data/songdo/videos/` (a symlink works), laid out as
`<day>/<drone>/<session>/<site><n>.MP4`; they are not released, and the record publishes the
prepared cases instead. Cases are written into `data/songdo/dynamic/` (see
[`songdo_static.md`](songdo_static.md#2-get-the-frames)). Commands run from the repository root.

## 1. Select the hovers

```bash
DAY=2022-10-06
stabilo-benchmark survey data/songdo/videos/$DAY/D*/*/[BCFGIJLMOPRSU][0-9]*.MP4 -c configs/songdo/dynamic/offline_masking.yaml \
  --anchor scan --max-per-video 1 -k 2 --controls 1 --group-pattern '_([A-Z])[0-9]+$' \
  --discard-frames --device mps -w data/songdo/survey
```

```
Read 831 segment(s) across 370 recording(s)
Splits (pool median): moving 0.675% of the frame, coherence 0.422; controls: stationarity >= 0.902 and static >= 1.408%

segment            cover%  static  moving  coher.  still  texture  margin  cell
-------------------------------------------------------------------------------
D10_PM5_U1@001170    3.00    0.66    1.94    0.80   0.25     3606    4.21  high moving / coherent
D3_PM3_F2@000259     1.94    0.29    1.43    0.72   0.17     4345    2.88  high moving / coherent
D9_AM3_S2@000300     3.96    1.76    1.89    0.22   0.48     4990    2.23  high moving / mixed
D1_PM1_B2@000240     3.29    1.49    1.48    0.22   0.50     4621    2.22  high moving / mixed
D7_PM1_P1@000300     1.81    1.53    0.17    0.74   0.90     4757    1.94  low moving / coherent
D3_AM1_M1@000237     0.92    0.69    0.18    0.59   0.79     4589    1.88  low moving / coherent
D7_PM5_P2@001200     2.35    1.95    0.24    0.23   0.89     3936    1.66  low moving / mixed
D8_PM3_M1@001162     1.77    1.39    0.26    0.25   0.84     4772    1.60  low moving / mixed
D9_AM1_R1@002100     2.02    1.94    0.00       -   1.00     4068    0.52  stationary control

Design:
  high moving / coherent   2/2  ok  (pool: 173)
  high moving / mixed      2/2  ok  (pool: 179)
  low moving / coherent    2/2  ok  (pool: 179)
  low moving / mixed       2/2  ok  (pool: 172)
  stationary control       1/1  ok  (pool: 56)
  72 candidate(s) unclassified: too few displaced objects for coherence
```

- `-c` sets the offsets read and the horizon: a scan point is kept only if the hover holds the
  config's sampling span (3600 frames for `offline_masking.yaml`) after it. `--horizon N` overrides it.
- `--anchor scan` reads a segment every `--scan-stride` frames (900, ~30 s), clear of the first and
  last seconds; `--max-per-video 1` keeps two cases from sharing a hover.
- Frames are decoded sequentially, as `prepare` decodes them, and annotated with the hbb2obb
  commands of the main README (`--device mps` on a Mac, where the default is the CPU).
  `--discard-frames` deletes a hover's frames once its segments are described; the labels and
  `STAGED.json`, which records each frame's SHA-256, stay. A repeated survey decodes the frames
  again, checks them against the record and reuses the labels, so changing a split costs a
  decoding pass, not an annotation pass; without `--discard-frames` it costs neither.
- `cover%` is total coverage; `static`, `moving` (displaced) and `coher.` are as above; `still` is
  stationarity. `margin` is the distance from the split boundaries in units of the pool's spread.
- Hovers named alike in different sessions (`A1.MP4`) are told apart by their path:
  `D1_AM1_A1`. `--group-pattern` reads the site from that name, and `--max-per-group` (default 1)
  keeps a cell from taking two hovers of one site.
- `SHORT` marks a cell the footage could not fill; it is not filled from another cell. Add the
  next day of the pool.
- Splits default to the pool median and are printed; pin them with `--moving-split` and
  `--coherence-split`. The survey also warns when an axis has no spread.
- `--anchor start` (the default) reads each recording from its first frame only.
- A large pool can be read in parallel: run the same command on disjoint subsets of the recordings,
  each with its own `--output <shard>.json`, then select from all of them at once with
  `stabilo-benchmark survey --pool <shard>.json ... -c <config>` and the selection options above.
  A shard read with another plan or other thresholds is refused, and the selection is the one a
  single survey over every recording makes.

The plan is written to `SURVEY.json`: every candidate with its per-offset fates, the thresholds it
was read with, the splits, the selection, and the `prepare` command for each selected segment.

## 2. Record the choice

Write the cell assignment to `configs/songdo/dynamic/cases.txt`. The manuscript's table of the cases
comes from the survey's own record:

```bash
stabilo-benchmark report data/songdo/survey/SURVEY.json   # -> tables/survey_cases.tex
```

## 3. Tune on separate cases

Part I's tuned ratio test and inlier threshold were selected on static scenes. Part II selects its
own, by the same procedure, on tuning cases from Part I's tuning sites (A, E, H, K, N, Q, T), so
no tuning case shares an intersection with an evaluation case. The survey picks them over the same
day, one per cell plus a control:

```bash
stabilo-benchmark survey data/songdo/videos/$DAY/D*/*/[AEHKNQT][0-9]*.MP4 -c configs/songdo/dynamic/offline_masking.yaml \
  --anchor scan --max-per-video 1 -k 1 --controls 1 --group-pattern '_([A-Z])[0-9]+$' \
  --discard-frames --device mps -w data/songdo/survey_tuning
```

They are listed in `configs/songdo/dynamic/cases_tuning.txt`, built like the evaluation cases (section 4) into
`$DATA/dynamic_tuning/`, and get no ground control points. First the reference estimator's own
two values are tuned in Part I's protocol, on the static tuning scenes with their known
homographies, at its feature budget and estimator settings and with its masking:

```bash
./scripts/songdo_static.sh oracle_tuning
```

Its mAA-optimal pair goes into the `oracle` preset, and `oracle --overwrite` recomputes every
case before any ground control point is placed. Then each pipeline is swept over Part I's grid,
unmasked, against the reference estimator, and pooled with equal weight per tuning case:

```bash
DATA=data/songdo
for CASE in $DATA/dynamic_tuning/*/; do
  stabilo-benchmark dynamic $CASE -c configs/songdo/dynamic/offline_tuning.yaml -v 1
done
stabilo-benchmark report $DATA/dynamic_tuning/*/results/offline_tuning
```

`./scripts/songdo_dynamic.sh offline_tuning` does the same over
`cases_tuning.txt`, and `realtime_tuning` repeats it at the real-time budget.
`<regime>_tuning_tuned_parameters.tex` gives each pipeline's selection beside its mAA at Part I's
values: Part I's values are kept unless a grid point beats them by more than the mAA equivalence
margin, with a dagger on a replacement at an end of its swept range (the report also warns: widen
the grid before adopting it). `<regime>_tuning_tuned_parameters.yaml` holds the selections as
per-detector `filter_ratio` and `ransac_epipolar_threshold` mappings; paste both into the
`pipeline` section of the regime's evaluation configs (`offline_masking.yaml` and
`offline_margin.yaml`, or `realtime_masking.yaml`). The presets keep Part I's values. To widen the
grid, edit the tuning config and rerun its campaign with `--regrid`: stored runs still in the grid
are kept and renumbered, and only the new grid points run.

## 4. Build, audit and benchmark each case

```bash
DATA=data/songdo
NAME=<day>_<segment name>            # e.g. 2022-10-06_D1_AM2_B3
CASE=$DATA/dynamic/$NAME
stabilo-benchmark prepare data/songdo/videos/<day>/<drone>/<session>/<hover>.MP4 -o $CASE \
  --reference <frame from SURVEY.json> --cell '<cell from SURVEY.json>' -c configs/songdo/dynamic/offline_masking.yaml
(cd $DATA && hbb2obb-detect dynamic/$NAME/images --model geotrax --no_confidence --save_provenance --device mps)
(cd $DATA && hbb2obb dynamic/$NAME/images --sam_models sam_l --imgsz 1024 --scale_factors 0.02 \
  --opening_kernel_percentage -0.1 --fragment_ratio 0.1 --confidence_dir --save_provenance --device mps)
stabilo-benchmark oracle  $CASE -c configs/songdo/dynamic/offline_masking.yaml
./scripts/draw_exclusion.sh $CASE   # buildings on the reference frame
stabilo-benchmark annotate-gcp $CASE             # reference by hand; two frames per audit bin from proposals
stabilo-benchmark annotate-gcp $CASE --repeat     # on two of the cases: unaided click repeatability
stabilo-benchmark audit   $CASE
stabilo-benchmark dynamic $CASE -c configs/songdo/dynamic/offline_masking.yaml -v 1
stabilo-benchmark audit   $CASE --results $CASE/results/offline_masking
stabilo-benchmark dynamic $CASE -c configs/songdo/dynamic/offline_margin.yaml -v 1
stabilo-benchmark dynamic $CASE -c configs/songdo/dynamic/realtime_masking.yaml -v 1
stabilo-benchmark audit   $CASE --results $CASE/results/realtime_masking
```

On the campaign machine, once every case is audited, `scripts/songdo_dynamic.sh` runs a campaign
over all cases of its list, logs to `data/songdo/logs/dynamic/<campaign>.log` and prints the pooled
`report` command. It refuses to start while a case is not audited, lacks its exclusion polygons, or
its cached reference estimates come from another oracle than the config names. `RESULTS_DIR` writes
each case's campaign to `$RESULTS_DIR/<case>/<campaign>` instead of into the case, to rerun a
published campaign and compare:

```bash
./scripts/songdo_dynamic.sh offline_masking    # then audit --results, per case
./scripts/songdo_dynamic.sh offline_margin
./scripts/songdo_dynamic.sh offline_repeats    # learned pipelines, 3 repeats, one case
./scripts/songdo_dynamic.sh realtime_masking   # then audit --results, per case
RESULTS_DIR=reruns/dynamic ./scripts/songdo_dynamic.sh offline_masking
```

`draw_exclusion.sh` opens the reference frame in LabelMe (`uv tool install labelme`); one polygon
per building, covering its roof and visible facades, saved as `$CASE/EXCLUSION.json`. The masking
and margin configs exclude those pixels from the evaluation region (`metrics.region.exclusion`):
above-ground structure has parallax once the platform drifts, which no homography removes. Poles,
signs and street trees are left in. Draw the polygons before any of the case's results are read,
so the region cannot be adjusted to them; the results record each file's SHA-256 and a resume
under different polygons refuses to start.

hbb2obb runs from the record root with relative paths because it writes the command into the
provenance files as typed; `--device` names the accelerator (`mps` on a Mac, `0` for the first CUDA
device) and is recorded in the provenance. The case's reference is the frame the survey selected
(`--reference`); `--cell` is recorded in `CASE.json` and with every measurement. Without `--frames`,
`annotate-gcp` audits two sampled frames in each of the config's `audit_bins` (1 to 3 frames, 4 to
31, 32 to 511, 512 to 3600), eight per case, so the offsets at which a moving group is still shared
with the reference are audited too, and `audit` reports the residual per bin. Run `audit` before
reading the candidates' numbers.

Ground control points are chosen in the reference frame by hand: at least twelve, on the road
surface, spread to the frame's periphery, clear of every vehicle by `mask_dilation_px` and outside
the exclusion polygons (`annotate-gcp` names any point that is not), and at least 65 pixels from the
frame edge, which a proposal needs. In each later frame every point opens on a proposal, a local
affine ECC fit of the point's reference neighborhood started from the platform motion, which
involves neither the reference estimator nor any candidate; Enter accepts it, a click or the arrow
keys correct it, `s` removes a point that is not visible. Correct a proposal that sits on a
different feature or visibly off it; a nudge below a pixel only adds click noise. A point without a
proposal opens near the reference estimator's prediction, displaced by a hidden offset, with nothing
placed. Each frame's file records its proposals, and `audit` reports how many were accepted,
corrected or placed by hand. The repeat session is placed without proposals, so the repeatability
measures unaided clicks. `annotate-gcp <case> --manual` places a case's audited frames again on
its reference points, every point by hand, into `gcp_manual/`: the comparison of unaided clicks
with the proposals.

## 5. Report

```bash
stabilo-benchmark report $CASE/results/offline_masking --kind dynamic   # one case
stabilo-benchmark report $DATA/dynamic/*/results/offline_masking --audit-case 2022-10-06_D9_AM3_S2    # all cases, pooled
stabilo-benchmark report $DATA/dynamic/*/results/realtime_masking --audit-case 2022-10-06_D9_AM3_S2   # the real-time regime
```

`--audit-case` names the case drawn in panels (a) and (b) of `figures/gcp_audit.pdf` (default: the
first audited case by name).

A single case gives the case's tables; it bolds nothing. Several directories pool the cases with
equal weight per case, in the order of their names whatever order they are given in, and add what
is read across cases:

- `*_dynamic.tex` and `*_gcp.tex`: every pipeline and masking mode, against the reference
  estimator and against the GCP-fitted transform, bolded under the analysis plan;
- `*_summary.pdf`: every pipeline and masking mode on one page, mAA against the reference
  estimator and the corner error against the GCP-fitted transform (median to 90th percentile),
  filled markers for the tie set of the analysis plan;
- `gcp_audit.tex`: the reference estimator's residual per case;
- `figures/cases.pdf`: each case's reference frame with every object outlined by what it did in
  the sampled frame closest to the survey's longest offset (displaced with the largest group that
  moved together, displaced otherwise, found where it was, not found), found again as the
  benchmark finds them, with the reference estimator as the platform motion (not drawn for tuning
  campaigns);
- `*_by_offset.tex`, `*_by_scenario.tex` (per survey cell) and `*_by_persistence.tex` (moving
  against stationary foreground), with the outcomes the masking contrast is read on;
- `*_contrasts.tex`: the planned masking contrasts (oriented against none, then horizontal against
  none and oriented against horizontal), each for the average pipeline and per pipeline, with the
  per-case sign count; the primary claim is oriented against none for the average pipeline;
  `*_mechanism.tex`: the coverage, persistence and motion-alignment contrasts, likewise for the
  average pipeline and per pipeline;
  `*_contrasts.csv`: every estimate, interval and p-value behind both;
- `*_benefit_strata.tex`: the oriented-against-none benefit of the average pipeline per offset bin,
  survey cell and persistence half, descriptive (no interval or test);
- `*_capture.tex`/`.csv`: consensus capture per pipeline and masking mode. Each pair records the
  share of the frame its moved objects cover and the largest group that moved together, found by
  the survey's matching with the reference estimator as the platform motion; a measurement records
  its capture fraction (0 when the estimate follows the ground where the group lies, 1 when it
  registers the group back onto where it was) and where its inliers lie. A measurement is captured
  at a capture fraction of 0.1 or more; the table lists the configurations captured at least once
  (the caption gives the largest capture fraction of the others, what the identity transform
  reaches and the smallest displacement of any group), counts the captured pairs and the cases they
  come from, and gives the inlier count and ratio of captured estimates with their rank among the
  same pipeline's estimates in the same case and offset bin; the CSV holds every configuration and
  the identity, and adds the other pairs' inlier statistics, the counts at capture fractions of
  0.25, 0.5 and 1, and the measurements whose corner error reaches 100 px;
- `*_capture_populations.tex`/`.csv`: what the capture count holds, for every configuration captured
  at least once: pulls within the survey's offsets, breakdowns (a corner error against the reference
  estimator of 100 px or more) and the rest, with the error at the group and its direction, read
  from the case frames, and the inliers against the same case's nearest offsets that are neither
  captured nor broken down;
- `*_capture_scatter.pdf`: every measurement's capture fraction against the offset, a panel per
  pipeline; `*_capture_by_offset.csv`/`.pdf`: the captured share per offset bin of every
  configuration captured at least once; `*_capture_example.pdf`: the captured unmasked measurement
  with the largest capture fraction within the survey's offsets, registered again beside the same
  pipeline under oriented-box masking (drawn only when the new registration reproduces the stored
  estimate), and `*_graphical_abstract.pdf`/`.png`, the same pair in a journal's graphical-abstract
  format (at least 1328 x 531 px).

A tuning campaign adds `<prefix>_tuning_surface.pdf` beside its tuning table, with the incumbent
values marked.

The margin campaign adds `offline_margin_benefit.tex`/`.pdf`/`.csv`: the reference-free gain over
no masking (read from the same case's `offline_masking` campaign, on the same pairs) against the share of the
frame the mask suppresses, per box geometry and margin. A campaign run with `repeats` above one adds
`*_repeat_spread.tex`, the run-to-run disagreement per pair against the case's audited floor.

## Layout

```
configs/songdo/dynamic/             one config per campaign; cases.txt (the cell assignment),
                                    cases_tuning.txt (the tuning cases)
scripts/songdo_dynamic.sh           one campaign over every case, with its preflight
scripts/draw_exclusion.sh           the exclusion polygons of a case
data/songdo/
  videos/                           source recordings (not released)
  survey/, survey_tuning/           SURVEY.json and shards/ (the parts pooled into it); the survey
                                    stages each hover's sampled frames and labels here, which are
                                    not released
  dynamic/<day>_<segment>/          one prepared case: images/, labels_*/, CASE.json, EXCLUSION.json,
                                    gcp/, oracle/, results/<campaign>/ (with GCP_SCORES.json after
                                    audit --results)
  dynamic_tuning/<day>_<segment>/   one tuning case, the same without gcp/
  logs/dynamic/                     one log per campaign (not released)
```
