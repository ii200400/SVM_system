import cv2
import numpy as np
import glob
# from get_intrinsic_params import get_intrinsic_params

def undistort(img, K, D, DIM):    
    # img = cv2.imread(fname)
    print(str(img))
    h,w = img.shape[:2]   

    ## new_K 설정 
    new_K = K.copy()
    new_K[0,0]=K[0,0]
    new_K[1,1]=K[1,1]

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)    
    
    return undistorted_img
    


# =============================================================================================================
# DIM, K, D = get_intrinsic_params((6,8), './data/simulation_intrinsic')

# DIM=(1280, 720)
# K=np.array([[416.0, 0.0, 640.0], [0.0, 416.0, 360.0], [0.0, 0.0, 1.0]])
# D=np.array([[6.9665838155561891e-02], [-9.0530701217360399e-02], [5.1127030009307156e-02], [-1.2596854649954789e-02]])

# images = glob.glob('./data/triangle/*.png')
# for fname in images:
#     img = cv2.imread(fname)
#     undistorted_img = undistort(img, K, D, DIM)
#     cv2.imshow(str(fname), undistorted_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

