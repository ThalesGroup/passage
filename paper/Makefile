# PASSAGE Paper - Makefile
# Build LaTeX document with latexmk

MAIN = main
LATEX = pdflatex
BIBTEX = biber
LATEXMK = latexmk

# Default target
all: $(MAIN).pdf

# Build PDF with latexmk (handles all dependencies automatically)
$(MAIN).pdf: $(MAIN).tex references.bib arxiv.sty
	$(LATEXMK) -pdf -pdflatex="$(LATEX) -interaction=nonstopmode -synctex=1" $(MAIN).tex

# Quick build (single pass, for drafts)
quick:
	$(LATEX) $(MAIN).tex

# Full manual build (if latexmk not available)
manual: $(MAIN).tex references.bib arxiv.sty
	$(LATEX) $(MAIN).tex
	bibtex $(MAIN)
	$(LATEX) $(MAIN).tex
	$(LATEX) $(MAIN).tex

# Clean auxiliary files
clean:
	$(LATEXMK) -c $(MAIN).tex
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.synctex.gz *.fdb_latexmk *.fls

# Clean everything including PDF
distclean: clean
	rm -f $(MAIN).pdf

# Watch mode (continuous compilation)
watch:
	$(LATEXMK) -pdf -pvc -pdflatex="$(LATEX) -interaction=nonstopmode -synctex=1" $(MAIN).tex

# Word count (approximate)
wordcount:
	@texcount -inc -total $(MAIN).tex

# Spell check
spellcheck:
	aspell --mode=tex check $(MAIN).tex

# Check for common LaTeX issues
lint:
	chktex $(MAIN).tex

.PHONY: all quick manual clean distclean watch wordcount spellcheck lint
