import cv2
# import datetime
import time

cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)
cap3 = cv2.VideoCapture(2)
cap4 = cv2.VideoCapture(3)

# cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
# cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
# cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
print('width1 :%d, height1 : %d' % (cap1.get(3), cap1.get(4)))
print('width2 :%d, height2 : %d' % (cap2.get(3), cap2.get(4)))
print('width3 :%d, height3 : %d' % (cap3.get(3), cap3.get(4)))
print('width4 :%d, height4 : %d' % (cap4.get(3), cap4.get(4)))

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

    

    frame += 1

    if(ret1) :
        # gray = cv2.cvtColor(frame1,  cv2.COLOR_BGR2GRAY)    # 입력 받은 화면 Gray로 변환

        cv2.imshow('cap1', frame1)    # 컬러 화면 출력
        
        # print("1: " + str(datetime.datetime.now()))
        # cv2.imshow('frame_gray', gray)    # Gray 화면 출력

    if(ret2) :
        # gray = cv2.cvtColor(frame2,  cv2.COLOR_BGR2GRAY)    # 입력 받은 화면 Gray로 변환

        cv2.imshow('cap2', frame2)    # 컬러 화면 출력
        # print("2: " + str(datetime.datetime.now()))

    if(ret3) :
        # gray = cv2.cvtColor(frame3,  cv2.COLOR_BGR2GRAY)    # 입력 받은 화면 Gray로 변환

        cv2.imshow('cap3', frame3)    # 컬러 화면 출력
        # print("3: " + str(datetime.datetime.now()))

    if(ret4) :
        # gray = cv2.cvtColor(frame4,  cv2.COLOR_BGR2GRAY)    # 입력 받은 화면 Gray로 변환

        cv2.imshow('cap4', frame4)    # 컬러 화면 출력
        
        # print("4: " + str(datetime.datetime.now()))

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
cap2.release()
cap3.release()
cap4.release()
cv2.destroyAllWindows()