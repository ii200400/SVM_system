import cv2
import numpy as np

def mouse_callback(event, x, y, flags, param): 
    print("마우스 이벤트 발생, x:", x ," y:", y) # 이벤트 발생한 마우스 위치 출력

screen = cv2.imread('C:/Users/multicampus/Desktop/S07P31D108/screen.jpg')
screen = cv2.resize(screen,(int(1536),int(860)))

# img = np.zeros((256, 256, 3), np.uint8)  # 행렬 생성, (가로, 세로, 채널(rgb)),bit)

cv2.namedWindow('image')  #마우스 이벤트 영역 윈도우 생성
cv2.setMouseCallback('image', mouse_callback)

while(True):

    cv2.imshow('image', screen)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:    # ESC 키 눌러졌을 경우 종료
        print("ESC 키 눌러짐")
        break
cv2.destroyAllWindows()


# src = src[70:640-130, 120:1440-120].copy()
# src = cv2.resize(src, (int(1200*1.5), int(440*1.5)))
