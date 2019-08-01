import cv2

image = cv2.imread('./data/math/KakaoTalk_Photo_2019-08-01-10-10-49.jpeg')
print(image.shape)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 10)
cv2.imshow('th', thresh)
cv2.waitKey(0)
