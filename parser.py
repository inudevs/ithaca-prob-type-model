import re
import cv2
import numpy as np
# import pytesseract
from PIL import Image
from unidecode import unidecode

# tessdata_config = '--tessdata-dir "./tessdata"'


# def ocr_text(filename):
#     return pytesseract.image_to_string(
#         Image.open(filename),
#         lang='kor',
#         config=tessdata_config)


# def get_score(filename):
#     text = unidecode(ocr_text(filename))
#     try:
#         score = re.search(r'(\d)[^\d]*$', text).group(1)
#     except AttributeError:
#         return None
#     return score

def parse_page(name):
    input_path = './images/{}.png'.format(name)
    img = cv2.imread(input_path)

    # 시험지 윗부분의 불필요한 제목, 선 등을 crop

    front = name.split('-')[4] == 0
    if front:  # 맨 앞 페이지인 경우
        x, y = (100, 390)
    else:
        x, y = (100, 250)
    image = img[y:y + 1200, x:x + 1200]
    img_height, img_width, _ = image.shape

    # 그레이스케일로 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 임계 처리
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # 팽창
    kernel = np.ones((18, 26), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)
    cv2.imshow('dilation', img_dilation)

    # 등치선 찾고 작은 순서대로 정렬
    ctrs, _ = cv2.findContours(
        img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(
        ctrs,
        key=lambda ctr: cv2.boundingRect(ctr)[0],
        reverse=True)
    # print(len(sorted_ctrs))

    result = image.copy()
    # 인식된 영역의 가로 길이는 자른 (전체 시험지 가로의 80%) / 2를 넘어야 함
    min_width = (img_width / 5) * 2
    # 인식된 영역의 세로 길이는 전체 시험지의 20%를 넘으면 안 됨
    max_height = (img_height / 5)

    # subject = ['korean', 'english', 'math'][int(name.split('-')[3])]
    subject = 'math'
    grade = ['_', 'first', 'second', 'third'][int(name.split('-')[2])]
    idx = 0

    for ctr in sorted_ctrs:
        # 경계 박스를 구함
        x, y, w, h = cv2.boundingRect(ctr)

        if not front:
            if y >= 25 and not y > (img_height / 2):
                continue
            # 중앙보다 위쪽인 박스 중 상단에 붙어 있는 박스가 아니면 패스
            # (시험지 윗부분의 보기 제거)

        # 박스 크기가 조건에 부합하지 않으면 패스
        if not w > min_width or h > max_height:
            continue

        # Getting ROI
        roi = image[y:y + h, x:x + w]
        cv2.rectangle(result, (x, y), (x + w, y + h), (90, 0, 255), 2)

        filename = './data/{}/{}/{}-{}.png'.format(subject, grade, name, idx)
        cv2.imwrite(filename, roi)
        idx += 1

        # print(get_score(filename))

    # cv2.imshow('result', result)
    # cv2.waitKey(0)


if __name__ == '__main__':
    parse_page('2008-03-1-2-7')
