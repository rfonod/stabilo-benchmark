# Songdo static campaign: Part I on the full scene set

The complete Part I recipe, from a directory of 4K aerial frames to the tables and figures. For a
quick run, use the sample in the [main README](../README.md#quick-start) instead.

## The scene set

29 frames of session 2022-10-07 PM4 from the hovering-drone data of the
[background paper](https://doi.org/10.1016/j.trc.2025.105205): the first frame of the first hover of
every drone-intersection pair flown in that session. Names are `2022-10-07_PM4_<site>_<drone>`,
sites `A`–`U` (`D` unused), drones `D1`–`D10`, all 3840×2160.

The frames, their annotations and the results of every campaign below are published in the data
record registered as `songdo` in `data/registry.yaml`,
[10.5281/zenodo.23623540](https://doi.org/10.5281/zenodo.23623540). Any directory of aerial frames
in the same layout works with this recipe.

### Tuning / evaluation split

| File | Scenes | Sites |
| --- | --- | --- |
| [`scenes_tuning.txt`](../configs/songdo/static/scenes_tuning.txt) | 9 | A, E, H, K, N, Q, T |
| [`scenes_eval.txt`](../configs/songdo/static/scenes_eval.txt) | 20 | B, C, F, G, I, J, L, M, O, P, R, S, U |

The split is by site, since two drones over one intersection see nearly the same scene: sites in
alphabetical order, every third one to tuning. Both files, in `configs/songdo/static/`, are passed
with `--scenes-list`.

## 1. Install

```bash
git clone https://github.com/rfonod/stabilo-benchmark.git && cd stabilo-benchmark
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
python -c "import cv2; print(cv2.__version__, [hasattr(cv2, f'{d}_create') for d in ('BRISK','KAZE','AKAZE')])"
# want: 4.x.y [True, True, True]
```

If a check prints `False`, see the OpenCV note in the [main README](../README.md#install).

## 2. Get the frames

From the repository root:

```bash
stabilo-benchmark fetch songdo        # or by hand: see the main README
export DATA=data/songdo                # where fetch extracts it
export SCENES=$DATA/static
ls $SCENES/images | head -3            # 2022-10-07_PM4_A_D1.jpg ...
```

To use a copy of the record kept elsewhere, or your own frames, link a directory in the same layout
at `data/songdo/static` or set `SCENES`.

```
$DATA/static/
  images/              29 frames, *.jpg
  labels_hbb/          step 3
  labels_obb/          step 3
  labels_confidence/   step 3
  PROVENANCE_hbb.txt   PROVENANCE_obb.txt
  results/             one directory per campaign of step 4 (published)
```

The published record already contains the annotations; step 3 is needed only for your own frames
or to regenerate them.

## 3. Annotate

From `$DATA`, with relative paths:

```bash
(cd $DATA && hbb2obb-detect static/images --model geotrax --no_confidence --save_provenance --device 0)
(cd $DATA && hbb2obb static/images --sam_models sam_l --imgsz 1024 --scale_factors 0.02 \
  --opening_kernel_percentage -0.1 --fragment_ratio 0.1 --confidence_dir --save_provenance --device 0)
```

The first run downloads the detector and SAM checkpoints (about 1.3 GB) into `$DATA/models/`; link
an existing directory there with `ln -s /path/to/models $DATA/models`. Use `--device mps` on Apple Silicon,
or drop the flag for CPU.

The HBB fallback share (objects where SAM found no mask and the horizontal box was kept):

```bash
cat "$SCENES"/labels_confidence/*.txt | awk '$1 == 0 {n++} END {print n+0, "of", NR}'
```

## 4. Run the campaigns

Each campaign is started with `scripts/songdo_static.sh`, which picks the config and the scene list
from the campaign name, checks the paths and annotations, writes the results to
`$SCENES/results/<campaign>/` and logs to `data/songdo/logs/static/<campaign>.log`. Run from the
repository root, in the order below, one campaign at a time so that timings are comparable. The
scenes are read from `$SCENES`, by default `data/songdo/static`. To rerun a published campaign
without overwriting it, set `RESULTS_DIR`, and compare the two directories:

```bash
RESULTS_DIR=reruns/static ./scripts/songdo_static.sh detectors      # -> reruns/static/detectors
```

To run a campaign:

```bash
screen -S budget -dm bash -c 'source .venv/bin/activate && ./scripts/songdo_static.sh budget'
screen -r budget          # reattach; Ctrl-A D detaches, screen -ls lists
```

| Campaign | Scenes | Runs | What it measures |
| --- | --- | ---: | --- |
| `budget` | tuning | 240 | accuracy against `downsample_ratio` × `max_features`, masking `none` and `obb` |
| `tuning` | tuning | 512 | `filter_ratio` × `ransac_epipolar_threshold` per preset, masking `none` |
| `oracle_tuning` | tuning | 70 | the same for Part II's reference estimator, at its own budget, masking `obb` |
| `threshold_scaling` | tuning | 160 | only if tuning did not run at the final `downsample_ratio` |
| `detectors` | eval | 30 | detector × masking |
| `detectors_default` | eval | 10 | the default thresholds, unmasked |
| `detectors_hover` | eval | 30 | detector × masking at hover-scale perturbation |
| `estimators` | eval | 108 | robust estimator × inlier threshold |
| `rotation` | eval | 7 × 9 | accuracy against in-plane rotation, one result directory per angle |

1. `budget`: choose the operating point (see `configs/songdo/static/budget.yaml`) and set
   `downsample_ratio` and `max_features` in `stabilo_benchmark/cfg/pipelines.yaml`. Part II's two
   budgets are points of the sweep: its cheapest point for the real-time regime and 0.5 / 2000 for
   the offline one (`configs/songdo/dynamic/{realtime,offline}_*.yaml`).
2. `tuning`: write each preset's mAA-optimal `filter_ratio` and `ransac_epipolar_threshold` into
   `stabilo_benchmark/cfg/pipelines.yaml` (`report` writes them to `tuned_parameters.tex`).
3. The evaluation campaigns.

Extra flags pass through to `stabilo-benchmark static`; `-r` resumes a stopped campaign:

```bash
./scripts/songdo_static.sh detectors -r
```

A changed pipeline grid runs only its new points with `--regrid`
(`./scripts/songdo_static.sh tuning --regrid`): stored runs still in the grid are
kept and renumbered, the others are deleted, and a change outside the `pipeline` section refuses to
run. A preset whose values change in `stabilo_benchmark/cfg/pipelines.yaml` reruns, and only it.

Run the learning-based presets on a GPU; stabilo's `device: auto` picks CUDA when available.

**Trial cache.** Each perturbed frame is built once per campaign under
`data/songdo/static/results/.cache/<campaign>/` and reused by every run; results are identical without
it. It takes about 25 MB per 4K scene and trial (about 7 GB for `tuning`, 50 GB for an evaluation
campaign at 100 trials), is skipped when it would not leave 10 GB free, and is deleted when the
campaign completes. `--no-trial-cache` turns it off.

## 5. Report

```bash
stabilo-benchmark report data/songdo/static/results/detectors  --kind static
stabilo-benchmark report data/songdo/static/results/estimators --kind static
```

Tables are written to `reports/tables/` and figures to `reports/figures/` by default. A listing
too long for a page goes beside its table as a CSV: `*_comparisons.tex` lists the comparisons behind
the mAA bold and `*_comparisons.csv` those behind every bolded column; `estimators_estimators.tex`
gives each estimator's mAA per threshold and `estimators_estimators.csv` every threshold's full row.
Figures beside the tables: `tuning_surface.pdf` (the tuning campaign: every grid point shaded by
its mAA gap to the pipeline's best, the kept and the untuned starting values marked), `estimators_estimators.pdf`
(mAA against the inlier threshold per estimator), `budget_compute.pdf` (the budget campaign), and
`detectors_trial.pdf` (a detector campaign that ran ORB unmasked and under OBB masks: trial 0 of the
scene with the most objects, registered again with the stored parameters, with ORB's inlier matches
and where its inliers lie with and without masks). Pipelines appear under display names (`RootSIFT`,
`DISK+LightGlue`); the CSVs keep the identifiers.

## Layout

```
configs/songdo/static/     one config per campaign; scenes_tuning.txt, scenes_eval.txt (the split)
scripts/songdo_static.sh   one campaign per invocation
data/songdo/static/        the scenes and results/<campaign>/ (from the record)
data/songdo/logs/static/   one log per campaign (not released)
```
