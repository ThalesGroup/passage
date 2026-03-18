MAIN := main
OUTDIR := ../../outputs/paper/neurips
LATEXMK := latexmk
LATEXMK_FLAGS := -lualatex -interaction=nonstopmode -halt-on-error -file-line-error
PDF := $(OUTDIR)/$(MAIN).pdf

.PHONY: all pdf clean distclean watch help

all: pdf

pdf: $(PDF)

$(PDF): $(MAIN).tex neurips_2026.sty checklist.tex references.bib
	mkdir -p $(OUTDIR)
	$(LATEXMK) $(LATEXMK_FLAGS) -outdir=$(OUTDIR) $(MAIN).tex

watch:
	mkdir -p $(OUTDIR)
	$(LATEXMK) $(LATEXMK_FLAGS) -pvc -outdir=$(OUTDIR) $(MAIN).tex

clean:
	$(LATEXMK) -c -outdir=$(OUTDIR) $(MAIN).tex || true

distclean: clean
	rm -f $(PDF)

help:
	@printf '%s\n' \
		'make -C paper/neurips        Build the PDF into outputs/paper/neurips/' \
		'make -C paper/neurips watch  Rebuild on changes' \
		'make -C paper/neurips clean  Remove aux files' \
		'make -C paper/neurips distclean  Remove aux files and PDF'
