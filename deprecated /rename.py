import os
import re

r = re.compile('\d{4}-\d{2}-\d-\d.pdf')

dir_path = './pdfs'
for pdf_name in os.listdir(dir_path):
    if r.match(pdf_name) is not None or pdf_name == '.gitkeep':
        continue
    year = pdf_name[:4]
    month = int(pdf_name[pdf_name.find('월') - 1])
    grade = 1
    if any(i in pdf_name for i in ['영어', '외국어']):
        subject = 1
    else: subject = 2 # 수학
    new_name = '{}-{:02d}-{}-{}.pdf'.format(year, month, grade, subject)
    print('{} <- {}'.format(new_name, pdf_name))
    os.rename(
        os.path.join(dir_path, pdf_name), 
        os.path.join(dir_path, new_name)
    )
