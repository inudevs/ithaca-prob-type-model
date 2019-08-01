from converter import convert_file
from parser import parse_page

names = []
for year in range(2008, 2020):
    for month in [3, 6, 9, 11]:
        for grade in [1, 2, 3]:
            for subject in [1]:  # math only
                names.append(
                    '{}-{:02d}-{}-{}'.format(year, month, grade, subject))
print(names)

for name in names:
    try:
        pages = convert_file(name)
        for page in pages:
            parse_page(page)
    except RuntimeError:
        print('[!] {}.pdf not found'.format(name))
        pass
