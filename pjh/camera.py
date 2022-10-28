import cv2
import datetime
import time
# from undistort import undistort
# from topview import topview
from undistop import undi_top
import numpy as np


cap1 = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(2)
cap3 = cv2.VideoCapture(3)
cap4 = cv2.VideoCapture(4)


# cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
# cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
print('width1 :%d, height1 : %d' % (cap1.get(3), cap1.get(4)))
# print('width2 :%d, height2 : %d' % (cap2.get(3), cap2.get(4)))
# print('width3 :%d, height3 : %d' % (cap3.get(3), cap3.get(4)))
# print('width4 :%d, height4 : %d' % (cap4.get(3), cap4.get(4)))

max_diff = 0
frame = 0
frame_starttime = time.time()

while(True):
    time1 = 0
    tiem4 = 0

    time1 = time.time()

    ret1, frame1 = cap1.read()    # Read 결과와 frame
    ret2, frame2 = cap2.read()
    ret3, frame3 = cap3.read()
    ret4, frame4 = cap4.read()
    # print(str(frame1))
    # print(str(frame2))
    # print(str(frame3))
    # print(str(frame4))

    frame += 1

    if(ret1) :
        cam1 = undi_top(frame1, 'right')         
        cv2.imshow('frame_1_right', cam1) 

    if(ret2) :
        cam2 = undi_top(frame2, 'front')   
        cv2.imshow('frame_2_front', cam2)
 
    if(ret3) :
        cam3 = undi_top(frame3, 'back')     
        cv2.imshow('frame_3_back', cam3)
 
    if(ret4) :
        cam4 = undi_top(frame4, 'left')       
        cv2.imshow('frame_4_left', cam4)

    if cv2.waitKey(1) == ord('c'):
        cv2.imwrite('right.png', cam1)
        cv2.imwrite('front.png', cam2)
        cv2.imwrite('back.png', cam3)
        cv2.imwrite('left.png', cam4)


    time4 = time.time()

    diff = time4-time1
    # print(time4-time1)
    if(max_diff < diff):
        max_diff = diff
        # print(max_diff)

    frame_endtime = time.time()
    # print(frame_endtime - frame_starttime)
    if(frame_endtime - frame_starttime > 1):
        # print(frame)
        frame_starttime = frame_endtime
        frame = 0

cap1.release()
# cap2.release()
# cap3.release()
# cap4.release()
cv2.destroyAllWindows()