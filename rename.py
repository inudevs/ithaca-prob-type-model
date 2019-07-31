import os

dir_path = './pdfs'
for pdf_name in os.listdir(dir_path):
    year = pdf_name[:4]
    month = int(pdf_name[pdf_name.find('월') - 1])
    grade = 1
    subject = 2
    new_name = '{}-{:02d}-{}-{}.pdf'.format(year, month, grade, subject)
    print('{} <- {}'.format(new_name, pdf_name))
    os.rename(
        os.path.join(dir_path, pdf_name), 
        os.path.join(dir_path, new_name)
    )
