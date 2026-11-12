# Data

```
data/
  README.md          committed  this file
  registry.yaml      committed  the Zenodo DOI, file list, pinned checksums and license of data/songdo/
  sample/            committed  quick-start data (~38 MB), a subset of songdo/
    README.md                   where each file comes from, and its license
    static/                     3 of Part I's 29 scenes + their labels                  -> Part I
    dynamic/<case>/             10 frames of one of Part II's cases + their labels,
                                ground control points and exclusion polygons            -> Part II
  songdo/            ignored    the paper's data, from `fetch songdo` (or a symlink to a local copy)
    static/                     Part I's 29 scenes; results/<campaign>/ for every Part I campaign
    dynamic/<case>/             Part II's nine cases; results/<campaign>/ inside each
    dynamic_tuning/<case>/      Part II's five tuning cases, with their tuning campaigns
    survey/, survey_tuning/     SURVEY.json, from which the cases were selected
    supplementary/              the CSV files the paper's captions name
    side_analyses/              the scripts of the paper's checks outside the benchmark
```

The two branches share one layout: a scene set is `images/` beside `labels_hbb/` and `labels_obb/`;
a case adds `CASE.json`, `EXCLUSION.json`, `gcp/` and, once `oracle` and `audit` have run,
`oracle/`; a campaign writes its results into `results/<campaign>/` inside the data it ran on
(`--results-dir` writes them elsewhere). Both branches hold one directory per case.

**`sample/`** is a smoke test: every command runs on it from a fresh clone with no download, video,
GPU or detector weights. Each of its files sits at the path it has in `songdo/`, and every file
but one case manifest is a byte-for-byte copy; three scenes and nine frame pairs are too few to
separate pipelines. [`sample/README.md`](sample/README.md) describes what it holds and where from.

Everything but the sample and the registry is ignored by git. The ignore rule is a whitelist, so
anything new written here stays untracked:

```gitignore
data/*
!data/README.md
!data/registry.yaml
!data/sample/
```

## Getting the paper's data

```bash
stabilo-benchmark fetch --list     # what is registered, what is already present
stabilo-benchmark fetch songdo     # download, verify, extract into data/songdo/ (the default dataset)
stabilo-benchmark fetch --check    # re-verify what is on disk, download nothing
```

Before downloading, `fetch` prints how much it will download and extract and the free space on the
destination volume. It refuses a download that would not fit and asks for confirmation before one
over 1 GB when run at a terminal; `--yes` skips the question. Downloads resume. Each file is
verified against the checksum pinned in `registry.yaml`, or against the one Zenodo reports when
none is pinned, and `data/songdo/FETCHED.json` records the record, its version, the files and
their digests. The archives are kept beside what they extract to, so `--check` can re-verify them.

`registry.yaml` pins one version of the record (its DOI, record id, and the size and MD5 of every
archive and of the license), the one this release's results come from. The record's README and
`SHA256SUMS` are verified against the checksum Zenodo reports, since the README can be updated on
the record with identifiers assigned after publication. A later version of the record arrives with
a release that updates the registry; `fetch` refuses to extract it over a copy of another version.

Other datasets in this layout can join the registry: publish them on Zenodo and open a pull request
that adds their entry to `registry.yaml`, whose header shows the fields and the checks to run.

The archives can also be downloaded by hand, all or only the part you need, and extracted into
`data/songdo/`; the main README's [Getting the Data](../README.md#getting-the-data) lists them. A
copy of the record kept elsewhere works the same once linked at `data/songdo`.

## Annotating images

The commands and settings are in the main README's [Annotating images](../README.md#annotating-images)
section. Run them from the root of the image set with relative paths, since hbb2obb records the
command and paths in its provenance files as typed.

## Building a dynamic case

A case is a directory of sampled frames, their annotations, and a `CASE.json` recording where they
came from. It can live anywhere:

```bash
# from a video: extract the sampled frames into <case>/images
stabilo-benchmark prepare <video> -o <case> -c configs/songdo/dynamic/offline_masking.yaml

# from frames that are already extracted (no video, no decoder); --frames adopts only the listed
# indices, and -c records the config's reference and offset bins
stabilo-benchmark prepare --from-frames <frames> -o <case> -c configs/songdo/dynamic/offline_masking.yaml
```

Then annotate `<case>/images` with the two hbb2obb commands. The sample's case was built with the
second form, from ten frames of one of the paper's cases, whose labels it copies
([`sample/README.md`](sample/README.md)).
The paper's cases were built from the source videos, which are not released;
[`docs/songdo_dynamic.md`](../docs/songdo_dynamic.md) is how.

## Licenses and attribution

`songdo/` is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) with its own
README and license, and the sample, part of it, under the same terms; their labels were written by
hbb2obb with the geo-trax detector, whose weights are also CC BY 4.0.
[`sample/README.md`](sample/README.md) gives the citations. These terms are separate from the
code's license.
