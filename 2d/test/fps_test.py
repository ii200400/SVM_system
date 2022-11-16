import cv2
import time

cap1 = cv2.VideoCapture(0)
# cap2 = cv2.VideoCapture(0)
# cap3 = cv2.VideoCapture(0)
# cap4 = cv2.VideoCapture(0)

cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap3.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# cap3.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap4.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# cap4.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

prev_time = 0
total_frames = 0
start_time = time.time()

while(True):

    curr_time = time.time()

    ret1, frame1 = cap1.read()    # Read 결과와 frame    
    ret2, frame2 = cap1.read()
    ret3, frame3 = cap1.read()
    ret4, frame4 = cap1.read()

    if cv2.waitKey(1) == ord('c'):
        cv2.destroyAllWindows()



    total_frames = total_frames + 1
    term = curr_time - prev_time
    fps = 1 / term

    cv2.imshow('ddd', frame1)
    prev_time = curr_time
    fps_string = f'term = {term:.3f},  FPS = {fps:.2f}'

    
    print(fps_string)
