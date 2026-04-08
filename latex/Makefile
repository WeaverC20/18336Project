MAIN = main
TEX = pdflatex
BIB = bibtex

.PHONY: pdf clean watch

pdf: $(MAIN).tex
	$(TEX) $(MAIN)
	$(BIB) $(MAIN)
	$(TEX) $(MAIN)
	$(TEX) $(MAIN)

clean:
	rm -f *.aux *.bbl *.bcf *.blg *.log *.out *.run.xml *.toc *.fls *.fdb_latexmk *.synctex.gz

watch:
	latexmk -pdf -pvc $(MAIN)
