#  https://moon-coco.tistory.com/m/entry/OpenCVCamera-Calibration%EC%B9%B4%EB%A9%94%EB%9D%BC-%EC%99%9C%EA%B3%A1-%ED%8E%B4%EA%B8%B0

import numpy as np
import cv2

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
wc = 8  ## 체스 보드 가로 패턴 개수 - 1
hc = 6  ## 체스 보드 세로 패턴 개수 - 1
objp = np.zeros((wc * hc, 3), np.float32)
objp[:, :2] = np.mgrid[0:wc, 0:hc].T.reshape(-1, 2)

objpoints = []
imgpoints = []

file = 'C:\\Users\\bit\\Downloads\\new test.png'  ## 체스 보드 이미지
dist_file = 'C:\\Users\\bit\\Downloads\\lane_test.png'  ## 왜곡된 이미지 (같은 화각으로 찍은 이미지)

img = cv2.imread(file)
_img = cv2.resize(img, dsize = (640, 480), interpolation = cv2.INTER_AREA)
# cv2.imshow('img', _img)
gray = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY) ## gray scale로 바꾸기
# cv2.waitKey(0)

ret, corners = cv2.findChessboardCorners(gray, (wc, hc), None)  ## 체스 보드 찾기
print(ret) 
## 만약 ret값이 False라면, 체스 보드 이미지의 패턴 개수를 맞게 했는지 확인하거나 (wc, hc) 
## 체스 보드가 깔끔하게 나온 이미지를 가져와야 한다

if ret == True:
    objpoints.append(objp)

    corners2 = cv2.cornerSubPix(gray, corners, (10, 10), (-1, -1), criteria) ## Canny86 알고리즘으로 도형이 겹치는 코너 점을 찾는다
    imgpoints.append(corners2)

    ## 찾은 코너 점들을 이용해 체스 보드 이미지에 그려넣는다
    img = cv2.drawChessboardCorners(_img, (wc, hc), corners2, ret)
    # cv2.imshow('img', img)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)  ## 왜곡 펴기
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1)
    ## mtx = getOptimalNewCameraMatrix parameter alpha 
    ## dist = Free scaling parameter 
    ## 4번째 인자 = between 0 (when all the pixels in the undistorted image are valid) and 1 (when all the source image pixels are retained in the undistorted image)
    ## 1에 가까울수록 왜곡을 펼 때 잘라낸 부분들을 더 보여준다
    ## 전체를 보고 싶다면 1, 펴진 부분만 보고 싶다면 0에 가깝게 인자 값을 주면 된다
    dst = cv2.undistort(img, mtx, dist) ## getOptimalNewCameraMatrix 함수를 쓰지 않은 이미지
    dst2 = cv2.undistort(img, mtx, dist, None, newcameramtx) ## 함수를 쓴 이미지
    cv2.imshow('num1', dst)
    cv2.imshow('num2', dst2)
    cv2.waitKey(0)

cv2.destroyAllWindows()