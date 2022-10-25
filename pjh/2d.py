import cv2
import datetime

cap = cv2.VideoCapture(1)
print(datetime.datetime.now())
num = 1
while True:
    ret, img = cap.read()
    cv2. imshow('camera', img)

    if cv2.waitKey(1) == ord('c'):
        print('chesschess' + str(num))
        img_captured = cv2.imwrite('chesschess_bottom' + str(num) +'.png', img)
        num += 1
    
    if cv2.waitKey(1) == ord('q'):
        img_captured = cv2.imwrite
        break

cap.release()
cv2.destroyAllWindows()

