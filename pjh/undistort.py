import cv2
import numpy as np
import glob
# from get_intrinsic_params import get_intrinsic_params

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
    


# =============================================================================================================
# DIM, K, D = get_intrinsic_params((5,8), 'pjh/data/intrinsic_cart')


side = 'left'
ratio = 2
images = glob.glob('pjh/1029/topview/' + side + '/*.png')


               
num = 1 
for fname in images:    
    img = cv2.imread(fname)
    undistorted_img = undistort(img, side, ratio)
    
    cv2.imshow('img', img)   
    cv2.imshow(side + '_cap_undistorted_img' +'.png', undistorted_img)    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(side + '_cap_undistorted.png', undistorted_img)
    num += 1

