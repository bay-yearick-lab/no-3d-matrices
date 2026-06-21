$pdflatex = 'pdflatex -synctex=1 -interaction=nonstopmode -file-line-error %O %S';
# build_paper.sh overrides this with bibtex8 when available.
$bibtex   = 'bibtex %O %B';
$pdf_mode = 1;
$out_dir  = '.';
$bibtex_use = 1;
$force_mode = 1;
