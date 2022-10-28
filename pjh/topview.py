import cv2
import numpy as np
import glob
from undistort import *
from cv2 import FONT_HERSHEY_COMPLEX



def topview(img_original):    
        
    chessboardx=5
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
    point2y=img.shape[0]/2+50
    point3x=img.shape[1]/2+50
    point3y=img.shape[0]/2
    point4x=img.shape[1]/2+50
    point4y=img.shape[0]/2+50
    print('ddddd')
    # ipm_pts = np.array([[448,609], [580,609], [580,741], [448,741]], dtype=np.float32)
    ipm_pts = np.array([[point2x,point2y], [point1x,point1y],[point4x,point4y],[point3x,point3y]], dtype=np.float32)
    ipm_matrix = cv2.getPerspectiveTransform(pts, ipm_pts)
    ipm = cv2.warpPerspective(img, ipm_matrix, (640,480))
    # ipm = cv2.rotate(ipm, cv2.ROTATE_180) 

    print('ipm')
    print(ipm_matrix)
    
    # ipm90 = cv2.rotate(ipm, cv2.ROTATE_90_CLOCKWISE) 
    # ipm180 = cv2.rotate(ipm, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # display (or save) images
    # cv2.imshow('img', img)
    # cv2.imshow('img', img)
   
    cv2.waitKey()

    return ipm


# =====================================================================

DIM=(640, 480)
K=np.array([[236.09690171418842, 0.0, 318.85338449602176], [0.0, 234.89387612174056, 249.52451910875553], [0.0, 0.0, 1.0]])
D=np.array([[-0.044587316407145215], [-0.006867926932128537], [0.0010003423736737584], [-6.98647544665307e-05]])

fname = 'pjh/data/stitch_4/back.png'
img = cv2.imread(fname)
# print(str(img))
# undistorted_img = undistort(img, K, D, DIM) 
# cv2.imwrite('left_undi.png', undistorted_img)

topview_img = topview(img)
# cv2.imshow('left_undi', undistorted_img)
cv2.imshow('back_top', topview_img) 
# cv2.imwrite('left_top.png', topview_img)
cv2.imshow('back_ori', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# cv2.waitKey(0)
# cv2.destroyAllWindows()

