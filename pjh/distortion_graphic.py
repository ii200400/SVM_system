import cv2
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

# 프로그램에서 얻은 camera parameters
fx = 422.753406
fy = 422.753483
cx = 627.723441
cy = 360.531994
k1 = -0.203159
k2 = 0.027893
k3 = 0
p1 = -0.001007
p2 = 0.001643
skew_c = 0

file = './data/simulation_intrinsic/pattern_009.png'

img_d = cv2.imread(file, cv2.IMREAD_GRAYSCALE)  ##1280-720
h, w = (img_d.shape[0], img_d.shape[1])
# cv2.imshow('ff', img_d)
img_u = np.zeros((h, w, 1), np.int8)  ## 검은색 이미지 생성
flag = 0

# img_u의 한 점 (xpu, ypu) normalize
for ypu in range(h): 
    for xpu in range(w):
        ynu = (ypu - cy)/fy
        xnu = (xpu - cx) / fx - (skew_c * ynu)
        ru = xnu**2 + ynu**2
        xnd = (1 + k1 * ru + k2 * ru**2 + k3 * ru**3) * xnu + 2 * p1 * xnu * ynu + p2 * (ru + 2 * xnu**2)
        ynd = (1 + k1 * ru + k2 * ru**2 + k3 * ru**3) * ynu + p1 * (ru + 2 * ynu**2) + 2 * p2 * xnu * ynu
        xpd = int(fx * (xnd + skew_c * ynd) + cx) 
        ypd = int(fy * ynd + cy)
        img_u[ypu][xpu] = img_d[ypd][xpd]
        cv2.imshow('black', img_u)
        key = cv2.waitKey(25)
        if key == 27:  ## ESC 누르면 종료
            flag = 1
            break
    if flag == 1:
        break

print(img_u)
cv2.imshow('dd', img_u)
cv2.waitKey(0)