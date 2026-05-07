# PASSAGE Dataset Terms

PASSAGE ships two distinct licensing surfaces:

- The generator code in this repository is released under the MIT license in `LICENSE`.
- The exported dataset artifacts inherit obligations from the upstream JAXA ALOS AW3D30 elevation product used to derive the terrain channel.

For dataset artifacts distributed on Hugging Face and in release bundles:

- Preserve attribution to JAXA ALOS World 3D - 30m (AW3D30) and any other provenance notices carried in the metadata.
- Do not represent the derived elevation-backed artifacts as public domain.
- Comply with the upstream AW3D30 usage terms for any redistribution or downstream reuse of the derived terrain data.
- Treat PASSAGE as a research benchmark, not a safety-of-life product. The dataset and generated paths do not certify operational deployment.

When in doubt:

- Use `LICENSE` for code questions.
- Use this file plus `DATASHEET.md` for dataset-distribution questions.
- Preserve both files when mirroring the release tree.
