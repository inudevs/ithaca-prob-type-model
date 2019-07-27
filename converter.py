import fitz


def convert(file_path, output_name, zoom=2.2):
    document = fitz.open(file_path)
    pages = document.pageCount
    print(pages)
    for page_num in range(0, pages):
        page = document.loadPage(page_num)
        mat = fitz.Matrix(zoom, zoom)
        pix = page.getPixmap(matrix=mat)
        pix.writePNG('{}-{}.png'.format(output_name, page_num))
    return True


def convert_file(name):
    convert('{}.pdf'.format(name), name)
    return name


if __name__ == '__main__':
    convert_file('2008-03-1-2')
