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
    new_K[0,0]=K[0,0]/2
    new_K[1,1]=K[1,1]/2

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)    
    
    return undistorted_img
    


# =============================================================================================================
# DIM, K, D = get_intrinsic_params((5,8), 'pjh/data/intrinsic_cart')


DIM=(640, 480)
K=np.array([[236.09690171418842, 0.0, 318.85338449602176], [0.0, 234.89387612174056, 249.52451910875553], [0.0, 0.0, 1.0]])
D=np.array([[-0.044587316407145215], [-0.006867926932128537], [0.0010003423736737584], [-6.98647544665307e-05]])


images = glob.glob('pjh/data/intrinsic_cart/*.png')
num = 1 
for fname in images:    
    img = cv2.imread(fname)
    undistorted_img = undistort(img, K, D, DIM)

    # cv2.imwrite(fname + '_top_undi' +'.png', undistorted_img)
    cv2.imshow('img', img)   
    cv2.imshow(fname + '_undistorted_img' +'.png', undistorted_img)    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#     num += 1

