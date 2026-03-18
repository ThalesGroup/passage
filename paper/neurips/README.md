# PASSAGE NeurIPS Paper

This folder contains a clean NeurIPS 2026 Evaluations & Datasets Track paper scaffold.

## Files

- `main.tex`: minimal anonymous LaTeX paper scaffold
- `references.bib`: bibliography placeholder
- `skeleton.md`: working paper outline, draft abstract, compliance notes, and missing-work checklist
- `neurips_2026.sty`: official NeurIPS 2026 style file
- `checklist.tex`: official NeurIPS checklist template

## Build

From the repository root:

```bash
make -C paper/neurips
```

The PDF is written to:

```text
outputs/paper/neurips/main.pdf
```

Useful commands:

```bash
make -C paper/neurips watch
make -C paper/neurips clean
make -C paper/neurips distclean
```

## Notes

- The LaTeX file intentionally stays lightweight and anonymous for submission.
- The substantive drafting plan currently lives in `skeleton.md`.
- The official NeurIPS checklist is included by default so the paper scaffold matches the required submission structure.
