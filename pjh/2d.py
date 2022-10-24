import cv2


cap = cv2.VideoCapture(1)


while True:
    ret, img = cap.read()
    cv2. imshow('camera', img)

    if cv2.waitKey(1) == ord('c'):
        img_captured = cv2.imwrite('img_captured.png', img)
    
    if cv2.waitKey(1) == ord('q'):
        img_captured = cv2.imwrite
        break

cap.release()
cv2.destroyAllWindows()

