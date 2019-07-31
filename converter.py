import fitz


def convert(file_path, output_name, zoom=2.2):
    document = fitz.open(file_path)
    pages = document.pageCount
    # print(pages)
    for page_num in range(pages):
        page = document.loadPage(page_num)
        mat = fitz.Matrix(zoom, zoom)
        pix = page.getPixmap(matrix=mat)
        pix.writePNG('./images/{}-{}.png'.format(output_name, page_num))
    print('[*] {} -> PNG'.format(file_path))
    return pages


def convert_file(name):
    pages = convert('{}.pdf'.format(name), name)
    return ['{}-{}'.format(name, p) for p in range(pages)]


if __name__ == '__main__':
    convert_file('2008-03-1-2')
