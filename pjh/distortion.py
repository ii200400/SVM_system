import cv2
import sys # assert cv2.__version__[0] == '3', 'The fisheye module requires opencv version >= 3.0.0'
import numpy as np
import os
import glob
# from regex import P

# 체커보드 
# CHECKERBOARD = (6,8)
CHECKERBOARD = (6,9)
'''
cv2.TERM_CRITERIA_EPS = 종료 조건의 타입으로 주어진 정확도(epsilon 인자)에 도달하면 반복을 중단
cv2.TERM_CRITERIA_MAX_ITER = max_iter 인자에 지정된 횟수만큼 반복하고 중단
cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER를 조합해 사용하면 두가지 조건 중 하나가 만족되면 반복이 중단
'''
subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

'''
Calibration Flag 조합
cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC = 외부는 내부 최적화를 반복할 때마다 다시 계산
cv2.fisheye.CALIB_CHECK_COND = 함수는 조건 번호의 유효성을 확인
cv2.fisheye.CALIB_FIX_SKEW = 스큐 계수(알파)는 0으로 설정되고 0을 유지
'''
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
_img_shape = None
objpoints = [] # 3d point in real world space1
imgpoints = [] # 2d points in image plane.

# 체커보드를 다각도에서 찍은 이미지들로부터 파라미터들을 찾는 과정
# images = glob.glob('./data/simulation_intrinsic/*.png')
images = glob.glob('./data/test_images/*.jpg')
print(images)
for fname in images:    
    img = cv2.imread(fname)
    if _img_shape == None:
        _img_shape = img.shape[:2] # image.shape(:2) => img.shape(height, width)
    else:
        assert _img_shape == img.shape[:2], "All i same size."    
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # 코너 찾기
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            # print('objp' + str(objp))
            cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
            imgpoints.append(corners)
            # print(imgpoints)
            N_OK = len(objpoints)
       
K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
rms, _, _, _, _ = \
    cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        K,
        D,
        rvecs,
        tvecs,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )
DIM = _img_shape[::-1]
print("Found " + str(N_OK) + " valid images for calibration")
print("DIM=" + str(_img_shape[::-1]))
print("K=np.array(" + str(K.tolist()) + ")")
print("D=np.array(" + str(D.tolist()) + ")")



## 왜곡 보정

# You should replace these 3 lines with the output in calibration step
## 제공 받은 DIM, K, D 파라미터 
# DIM=(1280, 720)
# K=np.array([[416.0, 0.0, 640.0], [0.0, 416.0, 360.0], [0.0, 0.0, 1.0]])
# D=np.array([[6.9665838155561891e-02], [-9.0530701217360399e-02], [5.1127030009307156e-02], [-1.2596854649954789e-02]])

### 함수 정의
def undistort(img_path):    
    img = cv2.imread(img_path)
    h,w = img.shape[:2]   

    ## new_K 설정 
    new_K = K.copy()
    new_K[0,0]=K[0,0]/3
    new_K[1,1]=K[1,1]/3

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)    
    cv2.imshow("undistorted", undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
print("K=np.array(" + str(K.tolist()) + ")")
print("D=np.array(" + str(D.tolist()) + ")")

## 체커보드 이미지들 확인
# images = glob.glob('./data/simulation_intrinsic/*.png')
images = glob.glob('./data/test_images/*.jpg')
for fname in images:
    undistort(fname)

## 이미지 확인 
# undistort('./data/triangle/triangle1.png')
    