import cv2
import numpy as np
import glob
from cv2 import FONT_HERSHEY_COMPLEX
import numpy as np

# 체커보드 

def get_intrinsic_params(CHECKERBOARD,image_folder):
    subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
    objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    _img_shape = None
    objpoints = [] # 3d point in real world space1
    imgpoints = [] # 2d points in image plane.

    # 체커보드를 다각도에서 찍은 이미지들로부터 파라미터들을 찾는 과정
    images = glob.glob(image_folder + '/*.png')

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
    DIM = _img_shape[::-1]
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    
    cv2.fisheye.calibrate(objpoints, imgpoints, gray.shape[::-1], K, D, rvecs, tvecs,  calibration_flags,
        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))
    
   
    print("Found " + str(N_OK) + " valid images for calibration")
    print("DIM=" + str(DIM))
    print("K=np.array(" + str(K.tolist()) + ")")
    print("D=np.array(" + str(D.tolist()) + ")")

    
    return DIM, K, D



# ==============================================================

get_intrinsic_params((6,8), './data/simulation_intrinsic')
    



