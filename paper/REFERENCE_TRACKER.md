# PASSAGE Dataset Paper Reference Tracker

This file stages dataset-track citation candidates for the PASSAGE paper.
It is deliberately small: the dataset-side workspace only needs a handful
of primary references for provenance, packaging, and release discipline.
The larger reference surface (learned planners, dense-vision analogs,
segmentation and regression families, ODD-coverage methodology, edge
deployment) is staged separately at
[`../../vision/paper/REFERENCE_TRACKER.md`](../../vision/paper/REFERENCE_TRACKER.md)
and lives in
[`../../paper/neurips/references.bib`](../../paper/neurips/references.bib).

## How To Use This File

- Add a reference here before wiring it into
  [`../../paper/neurips/references.bib`](../../paper/neurips/references.bib).
- Prefer primary sources; record where the citation would likely land so
  later integration stays cheap.
- **Do not commit `references.bib` changes from this workspace.** This
  file is a staging doc only; the paper repository owns the bibliography.
- If a staged reference becomes paper-ready, confirm its bibliographic
  details, decide whether it is a direct predecessor or an analog, record
  the target section in `main.tex`, and then update `references.bib` and
  the relevant prose together in the paper repo.

## Claim Families

### Dataset Release Artifacts

Potential manuscript use: justify the Datasheet appendix and the
trustworthiness language in §3, the Conclusion, and App. G.

Candidate references:

1. Gebru, T., et al. "Datasheets for Datasets." *Communications of the
   ACM*, 2021.
   Status: already present in `references.bib` as `gebru2021`.
   Supports: App. G mirror of `../DATASHEET.md`, §3 release paragraph,
   the Conclusion's trustworthy-evaluation framing.
   Likely manuscript landing: cited in §3 and App. G.

2. ML Commons Croissant metadata specification.
   Status: staging — Croissant ships with the dataset release tree but is
   not currently referenced from the paper body. Promote only if the
   manuscript adds a release-format paragraph.
   Supports: dataset-release tree only (not currently cited from
   `main.tex`).
   Likely manuscript landing: optional release-discipline footnote.

### Source Elevation Data

Potential manuscript use: cite the JAXA ALOS AW3D30 DSM as the source of
all elevation tiles in §3.1.

Candidate references:

1. Takaku, J., Tadono, T., Doutsu, M., Ohgushi, F., Kai, H. "Updates of
   'AW3D30' ALOS Global DSM with Other Open Access Datasets." ISPRS
   Archives, 2020.
   Status: already present in `references.bib` as `takaku2020`.
   Supports: §3.1 source-data paragraph, ODD Table~`tab:odd`,
   Conclusion footnote acknowledging JAXA's licence.
   Likely manuscript landing: cited as the source of truth for the
   elevation channel.

2. JAXA ALOS World 3D 30m product page (terms of use, dataset access).
   Status: already present in `references.bib` as `jaxa_aw3d30`.
   Supports: Conclusion footnote (license attribution to JAXA),
   Datasheet appendix.
   Likely manuscript landing: cited from §3.1 and the Conclusion
   footnote.

### Trustworthy-AI Standards (cross-reference)

The trustworthiness vocabulary in §3.3, §4.4, App. A, and App. F borrows
from two aviation-AI documents already cited in `references.bib`:
`easa_aicp_issue2` (EASA Concept Paper, Issue 02) and `sae_arp6983`
(SAE/EUROCAE ARP6983 / ED-324, draft). They live on the model-side
reference tracker and are listed here only as a pointer; do not duplicate
the entries in this file.

## Gaps To Fill Later

- A canonical citation for the Croissant specification (choose the
  primary reference document the MLCommons working group points at), if
  and when the manuscript starts referencing Croissant from the paper
  body.
- Optional: a primary reference for zstandard compression if the
  Parquet-packaging discussion expands beyond a one-line mention.
- Optional: a citation for the geographic hold-out practice in
  terrestrial remote-sensing benchmarks if §3.3.3 adds a comparison
  paragraph.

## Before Promoting A Reference Into The Paper

1. Confirm the exact bibliographic details from a primary source.
2. Decide whether the citation is a direct predecessor or background
   context.
3. Record the target section in `../../paper/neurips/main.tex`.
4. Update `../../paper/neurips/references.bib` and the relevant prose
   together in the paper repository. **Do not commit `.bib` edits from
   this workspace.**
