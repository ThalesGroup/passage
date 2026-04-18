# PASSAGE Dataset Paper Reference Tracker

This file stages dataset-track citation candidates for the PASSAGE paper. It
is deliberately small: the dataset-side workspace only needs a handful of
primary references for provenance, packaging, and release discipline. The
larger reference surface (learned planners, dense-vision analogs,
segmentation and regression families) is staged separately at
[`../../passage-vision/paper/REFERENCE_TRACKER.md`](../../passage-vision/paper/REFERENCE_TRACKER.md).

## How To Use This File

- Add a reference here before wiring it into
  `../../passage-paper/neurips/references.bib`.
- Prefer primary sources; record where the citation would likely land so
  later integration stays cheap.
- **Do not commit `references.bib` changes from this workspace.** This file
  is a staging doc only; the paper repository owns the bibliography.
- If a staged reference becomes paper-ready, confirm its bibliographic
  details, decide whether it is a direct predecessor or an analog, record
  the target section in `main.tex`, and then update
  `references.bib` and the relevant prose together in the paper repo.

## Claim Families

### Dataset Release Artifacts

Potential manuscript use: justify the Datasheet, Croissant metadata, and
trustworthiness language in Sec. 3, Sec. 7, and Appendices F / G.

Candidate references:

1. Gebru, T., et al. "Datasheets for Datasets." *Communications of the ACM*,
   2021.
   Status: already present in `references.bib` as `gebru2021`.
   Supports: Appendix F mirror of `../DATASHEET.md`, Sec. 3 release
   paragraph, Sec. 7 trustworthiness framing.
   Likely manuscript landing: cited in Sec. 3 and Appendix F.

2. ML Commons Croissant metadata specification.
   Status: staging — confirm exact citation form (report, standard, or
   primary paper) before promoting into `references.bib`.
   Supports: Appendix G summary of `../croissant.json`, Sec. 3 release
   paragraph.
   Likely manuscript landing: Appendix G.

### Source Elevation Data

Potential manuscript use: cite the JAXA ALOS AW3D30 DSM as the source of all
elevation tiles in Sec. 3.1.

Candidate references:

1. Takaku, J., Tadono, T., Doutsu, M., Ohgushi, F., Kai, H. "Updates of 'AW3D30'
   ALOS Global DSM with Other Open Access Datasets." ISPRS Archives, 2020.
   Status: already present in `references.bib` as `takaku2020`.
   Supports: Sec. 3.1 source-data paragraph, ODD Table 3, dataset-release
   Sec. 7.
   Likely manuscript landing: cited as source of truth for the elevation
   channel.

## Gaps To Fill Later

- A canonical citation for the Croissant specification (choose the primary
  reference document the MLCommons working group points at).
- Optional: a primary reference for zstandard compression if the Parquet
  packaging discussion in Sec. 3.4 expands beyond a one-line mention.
- Optional: a citation for the geographic hold-out practice in terrestrial
  remote-sensing benchmarks if Sec. 3.3 adds a comparison paragraph.

## Before Promoting A Reference Into The Paper

1. Confirm the exact bibliographic details from a primary source.
2. Decide whether the citation is a direct predecessor or background
   context.
3. Record the target section in `../../passage-paper/neurips/main.tex`.
4. Update `../../passage-paper/neurips/references.bib` and the relevant
   prose together in the paper repository. **Do not commit `.bib` edits
   from this workspace.**
