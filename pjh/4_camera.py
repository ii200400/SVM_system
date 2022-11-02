import cv2
import datetime
import time
import numpy as np
from cv2 import FONT_HERSHEY_COMPLEX

def undistort(img, side, ratio):    
    # img = cv2.imread(fname)
    if side == 'back' :
        DIM=(640, 480)
        K=np.array([[234.31775602644353, 0.0, 314.2513855472157], [0.0, 233.59962352968617, 238.87423326996165], [0.0, 0.0, 1.0]])
        D=np.array([[-0.045064119147887216], [-0.007831131934492355], [0.003854842129763424], [-0.0011869947699851067]])

    elif side == 'front':
        DIM=(640, 480)
        K=np.array([[236.00033166136186, 0.0, 320.80698188362993], [0.0, 235.65832997607666, 240.32680556708823], [0.0, 0.0, 1.0]])
        D=np.array([[-0.06666879404866978], [0.01978079736476898], [-0.011079087261112086], [0.001808135607204492]])

    elif side == 'right':
        DIM=(640, 480)
        K=np.array([[235.1982364007136, 0.0, 320.722613672998], [0.0, 234.7955596475616, 250.22150003050075], [0.0, 0.0, 1.0]])
        D=np.array([[-0.05024307206669386], [0.003083654157514218], [-0.0038559072176334217], [0.0006626971518285512]])

    elif side == 'left':
        DIM=(640, 480)
        K=np.array([[236.6660345650066, 0.0, 320.7924649951577], [0.0, 235.26512582520894, 240.40633958077817], [0.0, 0.0, 1.0]])
        D=np.array([[-0.03507719652282247], [-0.019779623502449793], [0.008738624015827543], [-0.0019232019838494904]])

    # DIM=(640, 480)
    # K=np.array([[236.6660345650066, 0.0, 322.7924649951577], [0.0, 235.26512582520894, 252.40633958077817], [0.0, 0.0, 1.0]])
    # D=np.array([[-0.03507719652282247], [-0.019779623502449793], [0.008738624015827543], [-0.0019232019838494904]])
        
    ## new_K 설정 
    new_K = K.copy()
    new_K[0,0]=K[0,0]/ratio
    new_K[1,1]=K[1,1]/ratio

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)    
    
    return undistorted_img

def topview(img_original, side):    
        
    chessboardx=5
    chessboardy=5
    CHECKERBOARD = (chessboardx,chessboardy)
    img = img_original
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    # print(ret)
    corner1x = corners[0][0][0]
    corner1y = corners[0][0][1]
    corner2x = corners[chessboardx-1][0][0]
    corner2y = corners[chessboardx-1][0][1]
    corner3x = corners[(chessboardx)*(chessboardy-1)][0][0]
    corner3y = corners[(chessboardx)*(chessboardy-1)][0][1]
    corner4x = corners[(chessboardx)*(chessboardy)-1][0][0]
    corner4y = corners[(chessboardx)*(chessboardy)-1][0][1]
    
    pts = np.array([[corner1x, corner1y], [corner2x, corner2y], [corner3x, corner3y], [corner4x, corner4y]], dtype=np.float32)
    idx=0
    for pt in pts:
        idx+=1
        cv2.circle(img, tuple(pt.astype(np.int)), 1, (0,0,255), 5)
        cv2.putText(img,str(idx), tuple(pt.astype(np.int)),FONT_HERSHEY_COMPLEX,1,(0,0,255))
    # compute IPM matrix and apply it

    if side == 'front':
        point1y=215
        point1x=295
        point2y=215
        point2x=345
        point3y=265
        point3x=295
        point4y=265
        point4x=345 

        # ipm_pts = np.array([[448,609], [580,609], [580,741], [448,741]], dtype=np.float32)
        ipm_pts = np.array([[point1x,point1y], [point2x,point2y], [point3x,point3y], [point4x,point4y]], dtype=np.float32)
        ipm_matrix = cv2.getPerspectiveTransform(pts, ipm_pts)

        size = (640, 480)
        # angle = cv2.ROTATE_90_CLOCKWISE
        ipm = cv2.warpPerspective(img, ipm_matrix, size)
        # ipm = cv2.rotate(ipm, angle) 

    elif side == 'back':
        point1y=215
        point1x=295
        point2y=215
        point2x=345
        point3y=265
        point3x=295
        point4y=265
        point4x=345 

        # ipm_pts = np.array([[448,609], [580,609], [580,741], [448,741]], dtype=np.float32)
        ipm_pts = np.array([[point4x,point4y], [point3x,point3y], [point2x,point2y], [point1x,point1y]], dtype=np.float32)
        ipm_matrix = cv2.getPerspectiveTransform(pts, ipm_pts)

        size = (640, 480)
        # angle = cv2.ROTATE_90_COUNTERCLOCKWISE
        ipm = cv2.warpPerspective(img, ipm_matrix, size)
        # ipm = cv2.rotate(ipm, angle) 

    elif side == 'right':
        
        point1x=215
        point1y=295
        point2x=215
        point2y=345
        point3x=265
        point3y=295
        point4x=265
        point4y=345  



        ipm_pts = np.array([[point3x,point3y], [point4x,point4y], [point1x,point1y], [point2x,point2y]], dtype=np.float32)  
        # ipm_pts = np.array([[480/2,0], [215,29], [480/2,640/2], [0,640/2]], dtype=np.float32)
        ipm_matrix = cv2.getPerspectiveTransform(pts, ipm_pts)
        
        size = (480, 640)
        # angle = cv2.ROTATE_180
        ipm = cv2.warpPerspective(img, ipm_matrix, size)
        # ipm = cv2.rotate(ipm, angle) 

    elif side == 'left' :
        point1x=215
        point1y=295
        point2x=215
        point2y=345
        point3x=265
        point3y=295
        point4x=265
        point4y=345 

        # ipm_pts = np.array([[448,609], [580,609], [580,741], [448,741]], dtype=np.float32)
        ipm_pts = np.array([[point3x,point3y], [point4x,point4y], [point1x,point1y], [point2x,point2y]], dtype=np.float32)
        ipm_matrix = cv2.getPerspectiveTransform(pts, ipm_pts)
        
        size = (480, 640)
        # angle = cv2.ROTATE_180
        ipm = cv2.warpPerspective(img, ipm_matrix, size)
        # ipm = cv2.rotate(ipm, angle) 

    print('=======================pts=========================')
    print(pts)

    print('=======================ipm_pts=======================')
    print(ipm_pts)    
    
    print('================== ipm_matrix ==============')
    print(ipm_matrix)

    return ipm

# =====================================================================

cap1 = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(2)
cap3 = cv2.VideoCapture(3)
cap4 = cv2.VideoCapture(4)

print(cap1)
print(cap2)
print(cap3)
print(cap4)

cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap3.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# cap3.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap4.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# cap4.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

print('width1 :%d, height1 : %d' % (cap1.get(3), cap1.get(4)))
# print('width2 :%d, height2 : %d' % (cap2.get(3), cap2.get(4)))
# print('width3 :%d, height3 : %d' % (cap3.get(3), cap3.get(4)))
# print('width4 :%d, height4 : %d' % (cap4.get(3), cap4.get(4)))

max_diff = 0
frame = 0
frame_starttime = time.time()
f1_num = f2_num = f3_num = f4_num = 1

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
        frame1_undi = undistort(frame1, 'front', 1.5)
        frame1_top = topview(frame1_undi, 'front')
        cv2.imshow('frame_1', frame1) 
        cv2.imshow('frame1_undi', frame1_undi) 
        cv2.imshow('frame1_top', frame1_top) 
        pass

    if(ret2) :
        frame2_undi = undistort(frame2, 'back', 1.5)
        frame2_top = topview(frame2_undi, 'back')
        cv2.imshow('frame_2', frame1) 
        cv2.imshow('frame2_undi', frame2_undi) 
        cv2.imshow('frame2_top', frame2_top) 
        pass

    if(ret3) :
        frame3_undi = undistort(frame3, 'left', 1.5)
        frame3_top = topview(frame3_undi, 'left')
        cv2.imshow('frame_3', frame3) 
        cv2.imshow('frame3_undi', frame3_undi) 
        cv2.imshow('frame3_top', frame3_top) 
        pass

    if(ret4) :
        frame4_undi = undistort(frame4, 'right', 1.5)
        frame4_top = topview(frame4_undi, 'right')
        cv2.imshow('frame_4', frame4) 
        cv2.imshow('frame4_undi', frame4_undi) 
        cv2.imshow('frame4_top', frame4_top) 
        pass

    if cv2.waitKey(1) == ord('a'):
        cv2.imwrite('frame1_' + str(f1_num) + '.png', frame1)
        print('captured =============> frame1_' + str(f1_num) + '.png') 
        f1_num += 1

    if cv2.waitKey(1) == ord('b'):
        cv2.imwrite('frame2_square_' + str(f2_num) + '.png', frame2)
        print('captured =============> frame2_square_' + str(f2_num) + '.png') 
        f2_num += 1
    
    if cv2.waitKey(1) == ord('c'):
        cv2.imwrite('frame3_square_' + str(f3_num) + '.png', frame3)
        print('captured =============> frame3_square_' + str(f3_num) + '.png') 
        f3_num += 1

    if cv2.waitKey(1) == ord('d'):
        cv2.imwrite('frame4_square_' + str(f4_num) + '.png', frame4)
        print('frame4_square_' + str(f4_num) + '.png')
        f4_num += 1

        
    if cv2.waitKey(1) == ord('z'):
        cv2.imwrite('captured =============> frame1_square_' + str(f1_num) + '.png', frame1)
        cv2.imwrite('captured =============> frame2_square_' + str(f2_num) + '.png', frame2)
        cv2.imwrite('captured =============> frame3_square_' + str(f3_num) + '.png', frame3)
        cv2.imwrite('captured =============> frame4_square_' + str(f4_num) + '.png', frame4)

        print('frame1_square_' + str(f1_num) + '.png')
        print('frame2_square_' + str(f2_num) + '.png')
        print('frame3_square_' + str(f3_num) + '.png')
        print('frame4_square_' + str(f4_num) + '.png') 
        f1_num += 1
        f2_num += 1
        f3_num += 1
        f4_num += 1

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
cap2.release()
cap3.release()
cap4.release()
cv2.destroyAllWindows()