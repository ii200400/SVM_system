import cv2
from cv2 import FONT_HERSHEY_COMPLEX
import numpy as np

chessboardx=7
chessboardy=7
CHECKERBOARD = (chessboardx,chessboardy)
img = cv2.imread('./data/topview/chess_1.jpg')
# point_list = []
# count = 0

# def mouse_callback(event, x, y, flags, param):
#     global point_list, count, img_original


#     # 마우스 왼쪽 버튼 누를 때마다 좌표를 리스트에 저장
#     if event == cv2.EVENT_LBUTTONDOWN:
#         print("(%d, %d)" % (x, y))
#         point_list.append((x, y))

#         print(point_list)
#         cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

# cv2.namedWindow('original')
# cv2.setMouseCallback('original', mouse_callback)

# while(True):

#     cv2.imshow("original", img)


#     height, width = img.shape[:2]


#     if cv2.waitKey(1)&0xFF == 32: # spacebar를 누르면 루프에서 빠져나옵니다.
#         break


# point1x=img.shape[1]/2
# point1y=img.shape[0]/2
# point2x=img.shape[1]/2
# point2y=img.shape[0]/2+100
# point3x=img.shape[1]/2+100
# point3y=img.shape[0]/2
# point4x=img.shape[1]/2+100
# point4y=img.shape[0]/2+100
# # print(point1x)
# #ipm_pts = np.array([[448,609], [580,609], [580,741], [448,741]], dtype=np.float32)
# pts = np.float32([list(point_list[0]),list(point_list[1]),list(point_list[2]),list(point_list[3])])
# ipm_pts = np.array([[point1x,point1y], [point2x,point2y],[point3x,point3y],[point4x,point4y]], dtype=np.float32)



# ipm_matrix = cv2.getPerspectiveTransform(pts, ipm_pts)
ipm_matrix = [[ 5.60305836e-02,  8.68421502e-01,  1.83593549e+02],
 [-3.18179595e-01,  4.37939116e-01,  4.07579443e+02],
 [-2.84493906e-05,  9.19147443e-04,  1.00000000e+00]]
print(str(ipm_matrix))



ipm = cv2.warpPerspective(img, ipm_matrix, img.shape[:2][::-1])

# ipm90 = cv2.rotate(ipm, cv2.ROTATE_90_CLOCKWISE) 
# # display (or save) images
# cv2.imshow('original', img)
# cv2.imshow('ipm', ipm90)
# cv2.waitKey()