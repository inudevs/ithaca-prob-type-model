import re
import cv2
import numpy as np
import pytesseract
from PIL import Image
from unidecode import unidecode

tessdata_config = '--tessdata-dir "./tessdata"'


def ocr_text(filename):
    return pytesseract.image_to_string(
        Image.open(filename),
        lang='kor',
        config=tessdata_config)


def get_score(filename):
    text = unidecode(ocr_text(filename))
    try:
        score = re.search(r'(\d)[^\d]*$', text).group(1)
    except AttributeError:
        return None
    return score


img = cv2.imread('./outfile.png')

# 시험지 윗부분의 불필요한 제목, 선 등을 crop

# 맨 앞 페이지인 경우
x, y = 100, 390
image = img[y:y + 1200, x:x + 1200]
img_height, img_width, _ = image.shape

# 그레이스케일로 변환
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 임계 처리
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# 팽창
kernel = np.ones((18, 30), np.uint8)
img_dilation = cv2.dilate(thresh, kernel, iterations=1)
# cv2.imshow('dilation', img_dilation)

# 등치선 찾고 작은 순서대로 정렬
ctrs, hier = cv2.findContours(
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

for idx, ctr in enumerate(sorted_ctrs):
    # 경계 박스를 구함
    x, y, w, h = cv2.boundingRect(ctr)

    # 박스 크기가 조건에 부합하지 않으면 패스
    if not w > min_width or h > max_height:
        continue

    # Getting ROI
    roi = image[y:y + h, x:x + w]
    # cv2.imshow('problem {}'.format(idx), roi)
    cv2.rectangle(result, (x, y), (x + w, y + h), (90, 0, 255), 2)

    filename = 'prob-{}.png'.format(idx)
    cv2.imwrite(filename, roi)

    print(get_score(filename))

cv2.imshow('result', result)
cv2.waitKey(0)
