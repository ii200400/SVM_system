import cv2
import numpy as np

def undi_top(img, side):    
    
    DIM=(640, 480)
    K=np.array([[236.09690171418842, 0.0, 318.85338449602176], [0.0, 234.89387612174056, 249.52451910875553], [0.0, 0.0, 1.0]])
    D=np.array([[-0.044587316407145215], [-0.006867926932128537], [0.0010003423736737584], [-6.98647544665307e-05]])

    ## new_K 설정 
    new_K = K.copy()
    new_K[0,0]=K[0,0]/1.5
    new_K[1,1]=K[1,1]/1.5

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)    
    

    ## top_view
    if side == 'front':
        ipm_matrix = np.array([[8.29672459e-02, -4.27604535e+00,  6.88445836e+02],[ 1.03220846e+00, -2.46295567e+00, -4.66454536e+01],[ 1.92678213e-04, -9.22097608e-03,  1.00000000e+00]])
        undi_top_img = cv2.warpPerspective(undistorted_img, ipm_matrix, (480, 640))
        undi_top_img = cv2.rotate(undi_top_img, cv2.ROTATE_90_CLOCKWISE) 

    elif side == 'back':
        ipm_matrix = np.array([[9.69380060e-02, -7.91495641e+00, -2.44294708e+03], [-8.73546826e+00, -1.41451627e+01,  3.00646339e+03], [-2.07438778e-05, -5.25476785e-02,  1.00000000e+00]])
        undi_top_img = cv2.warpPerspective(undistorted_img, ipm_matrix, (480, 640))
        undi_top_img = cv2.rotate(undi_top_img, cv2.ROTATE_90_CLOCKWISE) 

    elif side == 'right':
        ipm_matrix = np.array([[3.81299550e-02, -4.21019076e+01,  4.55115792e+03], [1.35157281e+01, -2.10835058e+01, -4.20165091e+03], [-8.14020486e-04, -7.86764882e-02, 1.00000000e+00]])
        undi_top_img = cv2.warpPerspective(undistorted_img, ipm_matrix, (640, 480))
        undi_top_img = cv2.rotate(undi_top_img, cv2.ROTATE_180) 

    elif side == 'left':
        ipm_matrix = np.array([[-1.57369308e-01, -7.88319450e+00,  1.11463903e+03], [ 2.23467981e+00, -4.29169598e+00, -4.76296400e+02], [-4.52448286e-04, -1.54697482e-02,  1.00000000e+00]])
        undi_top_img = cv2.warpPerspective(undistorted_img, ipm_matrix, (640, 480))
        # undi_top_img = cv2.rotate(undi_top_img, cv2.ROTATE_180)    


    return [undistorted_img, undi_top_img]




# fname = 'pjh/data/stitch_4/back.png'
# img = cv2.imread(fname)

# cv2.imshow('dd', undi_top(img,'front'))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

fname = 'pjh/data/stitch_4/right.png'
img = cv2.imread(fname)
# print(str(img))
# undistorted_img = undistort(img, K, D, DIM) 
# cv2.imwrite('left_undi.png', undistorted_img)
pics = undi_top(img, 'right')
undi = pics[0]
topv = pics[1]
# cv2.imshow('left_undi', undistorted_img)
cv2.imshow('undi', undi) 
cv2.imshow('topv', topv) 
# cv2.imwrite('left_top.png', topview_img)
cv2.imshow('back_ori', img)
cv2.waitKey(0)
cv2.destroyAllWindows()