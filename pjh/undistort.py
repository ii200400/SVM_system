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
    new_K[0,0]=K[0,0]/1
    new_K[1,1]=K[1,1]/1

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)    
    
    return undistorted_img
    


# =============================================================================================================
# DIM, K, D = get_intrinsic_params((6,8), './data/simulation_intrinsic')


DIM=(640, 480)
K=np.array([[234.13641315125017, 0.0, 321.8426830789856], [0.0, 232.78945753825985, 254.5448173478177], [0.0, 0.0, 1.0]])
D=np.array([[-0.036577984163583155], [-0.01560319609491953], [0.006981191481167077], [-0.0014375161895305334]])

# DIM=(640, 480)
# K=np.array([[234.2312699086491, 0.0, 321.8124086172564], [0.0, 232.88975260506393, 254.41608874811922], [0.0, 0.0, 1.0]])
# D=np.array([[-0.037386507399665383], [-0.014327744674023538], [0.00624722761473243], [-0.0013115237002334407]])

images = glob.glob('./data/chesschess/*.png')
num = 1 
for fname in images:    
    img = cv2.imread(fname)
    undistorted_img = undistort(img, K, D, DIM)

    # cv2.imwrite('chesschess' + str(num) +'.png', undistorted_img)
    cv2.imshow('img', img)   
    cv2.imshow('undistorted_img', undistorted_img)    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    num += 1


