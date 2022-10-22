import cv2
import numpy as np
import glob
import get_intrinsic_params
from undistort import *

point_list = []
count = 0

def mouse_callback(event, x, y, flags, param):
    global point_list, count, img_original  

    # 마우스 왼쪽 버튼 누를 때마다 좌표를 리스트에 저장
    if event == cv2.EVENT_LBUTTONDOWN:
        print("(%d, %d)" % (x, y))
        point_list.append((x, y))

        print(point_list)
        cv2.circle(img_original, (x, y), 3, (0, 0, 255), -1)


def topview(img_original):    
    
    cv2.namedWindow('original')
    cv2.setMouseCallback('original', mouse_callback)
    

    while(True):

        cv2.imshow("original", img_original)


        height, width = img_original.shape[:2]


        if cv2.waitKey(1)&0xFF == 32: # spacebar를 누르면 루프에서 빠져나옵니다.
            break


    # 좌표 순서 - 상단왼쪽 끝, 상단오른쪽 끝, 하단왼쪽 끝, 하단오른쪽 끝
    pts1 = np.float32([list(point_list[0]),list(point_list[1]),list(point_list[2]),list(point_list[3])])
    pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])

    print(pts1)
    print(pts2)

    M = cv2.getPerspectiveTransform(pts1,pts2)

    img_result = cv2.warpPerspective(img_original, M, (width,height))


    return img_result




# ====================================================================

DIM=(1280, 720)
K=np.array([[416.0, 0.0, 640.0], [0.0, 416.0, 360.0], [0.0, 0.0, 1.0]])
D=np.array([[6.9665838155561891e-02], [-9.0530701217360399e-02], [5.1127030009307156e-02], [-1.2596854649954789e-02]])

images = glob.glob('./data/triangle/*.png')
for fname in images:
    undistorted_img = undistort(fname, K, D, DIM)
    topview_img = topview(undistorted_img)
    cv2.imshow(str(fname), topview_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

