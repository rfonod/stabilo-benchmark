# Sample data

A subset of the paper's data record, so that both parts of the benchmark run from a fresh clone
without a download, video, GPU or detector weights. Each file sits at the path it has in the
record, and every file except one manifest is a byte-for-byte copy of the record's. It is a smoke
test, not evidence: three scenes and nine frame pairs exercise every code path but cannot rank
pipelines.

```
sample/
  static/                       Part I: three scenes
    images/                     3840×2160 JPEG
    labels_hbb/                 horizontal vehicle boxes (hbb2obb's default detector)
    labels_obb/                 their oriented conversions, row for row
    labels_confidence/          per-box conversion score; 0 would mark a box left as its HBB (none here)
    PROVENANCE_hbb.txt          the hbb2obb command, versions and checkpoint hashes behind the labels
    PROVENANCE_obb.txt
  dynamic/
    2022-10-06_D9_AM3_S2/       Part II: one prepared case, ten frames of one hover
      CASE.json                 source, reference frame, survey cell, offset bins and frame list
      EXCLUSION.json            polygons over buildings on the reference frame (evaluation region only)
      gcp/                      ground control points on the reference frame and eight later frames
      images/  labels_*/  PROVENANCE_*.txt      as in static/
```

## Running it

Part I, then its tables and figures:

```bash
stabilo-benchmark static configs/sample/static.yaml -v 1
stabilo-benchmark report data/sample/static/results/smoke
```

Part II, in the order the protocol prescribes: the ground control points bound the reference
estimator's error before any candidate is scored.

```bash
case=data/sample/dynamic/2022-10-06_D9_AM3_S2
stabilo-benchmark oracle  $case -c configs/sample/dynamic.yaml
stabilo-benchmark audit   $case
stabilo-benchmark dynamic $case -c configs/sample/dynamic.yaml -v 1
stabilo-benchmark audit   $case --results $case/results/smoke
stabilo-benchmark report  $case/results/smoke
```

Each part runs in about a minute on a laptop CPU. Campaigns write into the sample (`results/smoke/`,
and `oracle/` for the reference estimator), which git ignores; a second run of a campaign needs
`--overwrite` or `--resume`. `report` writes its tables and figures into `reports/` unless
`--output` names another directory.

The two configs in [`configs/sample/`](../../configs/sample/) are the paper's, restricted to ORB and
SIFT: `dynamic.yaml` is `configs/songdo/dynamic/offline_masking.yaml`, and `static.yaml` is
`configs/songdo/static/detectors.yaml` with 5 trials per scene instead of 100.

## Trying the annotation tools

The shipped points and polygons are files under git, so `git checkout data/sample` restores them
after an experiment.

```bash
# A second, independent session on the audited frames, written to gcp_repeat/; audit then reports
# the annotator's repeatability beside the estimator's error (needs an OpenCV build with a GUI)
stabilo-benchmark annotate-gcp $case --repeat
stabilo-benchmark audit $case

# Reopen and edit the exclusion polygons in LabelMe (uv tool install labelme)
./scripts/draw_exclusion.sh $case
```

## Where it comes from

Both parts come from the paper's data record, which holds frames of the KAIST–EPFL multi-drone
experiment over the Songdo International Business District, South Korea: 4K video at 29.97 fps from
drones hovering 140 to 150 m up.

**`static/`** holds three of the record's 29 Part I scenes, frames of session PM4 on 2022-10-07:
the scenes with the fewest, the median and the most detected vehicles. A name encodes the date, the
session, the intersection (site) and the drone.

| Scene                  | Site | Drone | Vehicles |
| ---------------------- | ---- | ----- | -------- |
| `2022-10-07_PM4_Q_D8`  | Q    | D8    | 16       |
| `2022-10-07_PM4_S_D10` | S    | D10   | 78       |
| `2022-10-07_PM4_L_D6`  | L    | D6    | 152      |

**`dynamic/`** holds ten of the 136 frames of the record's case `2022-10-06_D9_AM3_S2` (drone D9
over site S2 in session AM3 on 2022-10-06), the case that illustrates the paper's audit figure: the
reference frame, 300; the eight frames its ground control points were placed on, two in each audit
offset bin, at offsets 1, 3, 13, 22, 87, 213, 1166 and 2016 frames; and frame 302, so that every
offset from 1 to 3 frames is present. `CASE.json` is the one file not copied: `prepare
--from-frames` wrote it when it adopted the ten frames, with the case's reference frame, frame rate
and survey cell and the offset bins of `configs/sample/dynamic.yaml`.

**Labels** were written by [hbb2obb](https://github.com/rfonod/hbb2obb) with the commands of the
main README's [Annotating images](../../README.md#annotating-images) section, over the record's
full image sets; the provenance files describe those runs. The horizontal boxes come from hbb2obb's
default `geotrax` detector, whose class IDs are 0 car, 1 bus, 2 truck and 3 motorcycle; the oriented
boxes are their SAM conversions, one per horizontal box in the same order. Coordinates are absolute
pixels.

With the record fetched (`stabilo-benchmark fetch songdo`), `pytest tests/test_samples.py` also
checks every copied file against the record's, byte for byte. The paper's `detectors` and
`offline_masking` campaigns also measure the sample's trials on `2022-10-07_PM4_S_D10` and
`2022-10-07_PM4_L_D6` and its nine frame pairs, with the same settings and seeds. On Linux x86_64,
where those campaigns ran, a run on the sample reproduces their estimates bit for bit. OpenCV's
feature detectors are not bit-reproducible across CPU architectures, so on another platform the
numbers can differ slightly from the record's.

## License and attribution

The sample is part of the data record and shares its license, the Creative Commons Attribution 4.0
International license ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)), which is
separate from the code's. Please cite the record and the experiment it comes from:

- Fonod, R., Geroliminis, N. (2026). *Accompanying data for the paper "Consensus Capture in
  Reference-Based UAV Video Stabilization: A Benchmark on Dynamic Urban Scenes"* (v1). Zenodo.
  [10.5281/zenodo.23623540](https://doi.org/10.5281/zenodo.23623540)
- Fonod, R., Cho, H., Yeo, H., Geroliminis, N. (2025). *Advanced computer vision for extracting
  georeferenced vehicle trajectories from drone imagery.* Transportation Research Part C, 178,
  105205. [10.1016/j.trc.2025.105205](https://doi.org/10.1016/j.trc.2025.105205)

The horizontal boxes were detected with the weights of [Geo-trax](https://github.com/rfonod/geo-trax)'s
vehicle detector, released under [CC BY 4.0](https://huggingface.co/rfonod/geo-trax): cite Geo-trax
and the background paper above when you report results produced with them.
