import numpy as np
import cv2

img = np.zeros((296*2,260*2, 3), np.uint8)
front_img = cv2.imread('pjh/1029/topview_undi_top_done_v2/front_undi_top_v2.png')
back_img = cv2.imread('pjh/1029/topview_undi_top_done_v2/back_undi_top_v2.png')
left_img = cv2.imread('pjh/1029/topview_undi_top_done_v2/left_undi_top_v2.png')
right_img = cv2.imread('pjh/1029/topview_undi_top_done_v2/right_undi_top_v2.png')
cv2.imshow('ddd', img)
# cv2.imshow('dddd', right_img)
cv2.waitKey(0)


''' 
실제 : 100 + 60 + 100 / 100 + 96 + 100

픽셀 : 520 / 592
'''