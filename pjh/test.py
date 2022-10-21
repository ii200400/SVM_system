import numpy as np
import cv2


point_list = []
count = 0

def mouse_callback(event, x, y, flags, param):
    global point_list, count, img_original


    # 마우스 왼쪽 버튼 누를 때마다 좌표를 리스트에 저장
    if event == cv2.EVENT_LBUTTONDOWN:
        print("(%d, %d)" % (x, y))
        point_list.append((x, y))

        print(point_list)
        cv2.circle(img_original, (x, y), 3, (0, 0, 255), -1)



cv2.namedWindow('original')
cv2.setMouseCallback('original', mouse_callback)

# 원본 이미지
img_original = cv2.imread('./data/topview/exam.gif')
img = cv2.imread('./data/topview/exam.gif')

if str(img).lower().endswith('exam.gif'):
    gif = cv2.VideoCapture(img)
    ret, img_original = gif.read()  # ret=True if it finds a frame else False.

print(img_original)


# gif = cv2.VideoCapture(img)
# # ret, img_original = gif.read()  # ret=True if it finds a frame else False.
# while True:
#     ret, img_original = gif.read()
#     pts1 = np.float32([[160, 110],[410, 110],[0, 180],[550, 180]])
# # height, width = img_original.shape[:2]
# # pts1 = np.float32([list(point_list[0]),list(point_list[1]),list(point_list[2]),list(point_list[3])])
#     pts2 = np.float32([[0,0],[613,0],[0,304],[613,304]])

#     print(pts1)
#     print(pts2)

#     M = cv2.getPerspectiveTransform(pts1,pts2)

#     img_result = cv2.warpPerspective(img_original, M, (613,304))

#     cv2.imshow("original", img_original)
#     cv2.imshow("result1", img_result)
    
#     if cv2.waitKey(1)&0xFF == 32:
#         break



while(True):

    cv2.imshow("original", img_original)


    height, width = img_original.shape[:2]


    if cv2.waitKey(1)&0xFF == 32: # spacebar를 누르면 루프에서 빠져나옵니다.
        break


# 좌표 순서 - 상단왼쪽 끝, 상단오른쪽 끝, 하단왼쪽 끝, 하단오른쪽 끝

#  [634.   0.]
#  [  0. 478.]
#  [634. 478.]
# pts1 = np.float32([[160, 110],[410, 110],[0, 180],[550, 180]])
# height, width = img_original.shape[:2]
pts1 = np.float32([list(point_list[0]),list(point_list[1]),list(point_list[2]),list(point_list[3])])
pts2 = np.float32([[0,0],[613,0],[0,304],[613,304]])

print(pts1)
print(pts2)

M = cv2.getPerspectiveTransform(pts1,pts2)

img_result = cv2.warpPerspective(img_original, M, (613,304))

cv2.imshow("original", img_original)
cv2.imshow("result1", img_result)
cv2.waitKey(0)
cv2.destroyAllWindows()