import cv2
import numpy as np
import glob
from undistort import *

point_list = []
count = 0
img_original = None

def mouse_callback(event, x, y, flags, param):
    global point_list, count, img_original  

    # 마우스 왼쪽 버튼 누를 때마다 좌표를 리스트에 저장
    if event == cv2.EVENT_LBUTTONDOWN:
        print("(%d, %d)" % (x, y))
        point_list.append((x, y))

        print(point_list)
        cv2.circle(img_original, (x, y), 3, (0, 0, 255), -1)


def topview(img_original):    
    
    # cv2.namedWindow('original')
    # cv2.setMouseCallback('original', mouse_callback)
    

    # while(True):

    #     cv2.imshow("original", img_original)


    #     height, width = img_original.shape[:2]


    #     if cv2.waitKey(1)&0xFF == 32: # spacebar를 누르면 루프에서 빠져나옵니다.
    #         break


    # # 좌표 순서 - 상단왼쪽 끝, 상단오른쪽 끝, 하단왼쪽 끝, 하단오른쪽 끝
    # pts1 = np.float32([list(point_list[0]),list(point_list[1]),list(point_list[2]),list(point_list[3])])
    # pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])

    # print(pts1)
    # print(pts2)

    # M = cv2.getPerspectiveTransform(pts1,pts2)

    # img_result = cv2.warpPerspective(img_original, M, (width,height))


    # return img_result

    # ========================
    import cv2
    from cv2 import FONT_HERSHEY_COMPLEX
    import numpy as np
    chessboardx=8
    chessboardy=5
    CHECKERBOARD = (chessboardx,chessboardy)
    img = img_original
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    print(ret)
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

    point1x=img.shape[1]/2
    point1y=img.shape[0]/2
    point2x=img.shape[1]/2
    point2y=img.shape[0]/2+100
    point3x=img.shape[1]/2+100
    point3y=img.shape[0]/2
    point4x=img.shape[1]/2+100
    point4y=img.shape[0]/2+100
    print(point1x)
    #ipm_pts = np.array([[448,609], [580,609], [580,741], [448,741]], dtype=np.float32)
    ipm_pts = np.array([[point1x,point1y], [point2x,point2y],[point3x,point3y],[point4x,point4y]], dtype=np.float32)
    ipm_matrix = cv2.getPerspectiveTransform(pts, ipm_pts)
    ipm = cv2.warpPerspective(img, ipm_matrix, img.shape[:2][::-1])

    print(ipm_matrix)
    ipm90 = cv2.rotate(ipm, cv2.ROTATE_90_CLOCKWISE) 
    ipm180 = cv2.rotate(ipm, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # display (or save) images
    # cv2.imshow('img', img)
    # cv2.imshow('img', img)
   
    cv2.waitKey()

    return ipm90


# =====================================================================

DIM=(640, 480)
K=np.array([[234.2312699086491, 0.0, 321.8124086172564], [0.0, 232.88975260506393, 254.41608874811922], [0.0, 0.0, 1.0]])
D=np.array([[-0.037386507399665383], [-0.014327744674023538], [0.00624722761473243], [-0.0013115237002334407]])

images = glob.glob('./data/bottom/*.png')
for fname in images:
    img = cv2.imread(fname)
    undistorted_img = undistort(img, K, D, DIM)
    topview_img = topview(undistorted_img)
    cv2.imshow(str(fname), undistorted_img)
    cv2.imshow(str(fname), topview_img)
    cv2.imshow(str(fname), img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

