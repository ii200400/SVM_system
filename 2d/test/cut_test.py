import cv2
import time

src = cv2.imread('C:/Users/multicampus/Desktop/S07P31D108/2d/data/extrinsic/front_undi.png')
cap1 = cv2.VideoCapture(0)
while True:
    ret1, frame1 = cap1.read()
# 이미지를 자른다.
dst = src[40:660, 100:1180].copy()
dst = cv2.resize(dst,(int(1080*0.8),int(620*0.8)))




# cv2.namedWindow('screen',cv2.WINDOW_NORMAL)
# cv2.setWindowProperty('screen', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


# cv2.imshow('source', src)
cv2.imshow('screen', dst)

cv2.waitKey(0)
cv2.destroyAllWindows()


