import fitz
pdffile = "2008-03-1-2.pdf"
doc = fitz.open(pdffile)
page = doc.loadPage(0)

zoom = 2.2
mat = fitz.Matrix(zoom, zoom)

pix = page.getPixmap(matrix = mat)

output = "outfile.png"
pix.writePNG(output)
