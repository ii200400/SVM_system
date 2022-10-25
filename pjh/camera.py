import cv2
import datetime
import time
from undistort import undistort
import numpy as np

cap1 = cv2.VideoCapture(1)
print(cap1)
# cap2 = cv2.VideoCapture(3)
# cap3 = cv2.VideoCapture(4)
# cap4 = cv2.VideoCapture(5)

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
    # ret2, frame2 = cap2.read()
    # ret3, frame3 = cap3.read()
    # ret4, frame4 = cap4.read()
    print(str(frame1))

    frame += 1

    if(ret1) :
        # gray = cv2.cvtColor(frame1,  cv2.COLOR_BGR2GRAY)    # 입력 받은 화면 Gray로 변환       
        DIM=(640, 480)
        # K=np.array([[240.0605271313352, 0.0, 321.66080907924186], [0.0, 238.81254131350977, 249.0682634019097], [0.0, 0.0, 1.0]])
        # D=np.array([[0.04946423066512006], [-0.3797668812210116], [0.5721825992005147], [-0.31702418953372935]])
    

        DIM=(640, 480)
        DIM=(640, 480)
        K=np.array([[286.29851424, 0.0, 312.01931813], [0.0, 282.83475255, 276.39723983], [0.0, 0.0, 1.0]])
        D=np.array([[0.06039191, -0.01057358, -0.00032372, -0.0051587 ]])
        undisdort_frame = undistort(frame1.copy(), K, D, DIM)
        
        cv2.imshow('frame', frame1)
        cv2.imshow('undisdort_frame', undisdort_frame)    # 컬러 화면 출력
        






        # print("1: " + str(datetime.datetime.now()))
        # cv2.imshow('frame_gray', gray)    # Gray 화면 출력

    # if(ret2) :
    #     # gray = cv2.cvtColor(frame2,  cv2.COLOR_BGR2GRAY)    # 입력 받은 화면 Gray로 변환

    #     cv2.imshow('frame2_color', frame2)    # 컬러 화면 출력
    #     # print("2: " + str(datetime.datetime.now()))

    # if(ret3) :
    #     # gray = cv2.cvtColor(frame3,  cv2.COLOR_BGR2GRAY)    # 입력 받은 화면 Gray로 변환

    #     cv2.imshow('frame3_color', frame3)    # 컬러 화면 출력
    #     # print("3: " + str(datetime.datetime.now()))

    # if(ret4) :
    #     # gray = cv2.cvtColor(frame4,  cv2.COLOR_BGR2GRAY)    # 입력 받은 화면 Gray로 변환

    #     cv2.imshow('frame4_color', frame4)    # 컬러 화면 출력
        
    #     # print("4: " + str(datetime.datetime.now()))

    if cv2.waitKey(1) == ord('q'):
        break

    time4 = time.time()

    diff = time4-time1
    # print(time4-time1)
    if(max_diff < diff):
        max_diff = diff
        print(max_diff)

    frame_endtime = time.time()
    # print(frame_endtime - frame_starttime)
    if(frame_endtime - frame_starttime > 1):
        print(frame)
        frame_starttime = frame_endtime
        frame = 0

cap1.release()
# cap2.release()
# cap3.release()
# cap4.release()
cv2.destroyAllWindows()