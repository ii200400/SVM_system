import cv2
import numpy as np
def undistort_left(img, ratio):
    DIM=(1280, 720)
    K=np.array([[486.43710381577273, 0.0, 643.0021325671074], [0.0, 485.584911786959, 402.9808925210084], [0.0, 0.0, 1.0]])
    D=np.array([[-0.06338733272909226], [-0.007861033496168955], [0.005073683389947028], [-0.0010639404289377306]])

    ## new_K 설정 
    new_K = K.copy()
    new_K[0,0]=K[0,0]/ratio
    new_K[1,1]=K[1,1]/ratio

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, DIM, cv2.CV_16SC2)

    # print(map1, map2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img

def undistort(img, ratio):
    DIM=(1280, 720)
    K=np.array([[455.8515274977241, 0.0, 655.7621645964248], [0.0, 455.08604281075947, 367.3548823943176], [0.0, 0.0, 1.0]])
    D=np.array([[-0.02077978156022359], [-0.02434621475644252], [0.009725498728069807], [-0.0018108318059442028]])
    

    ## new_K 설정 
    new_K = K.copy()
    new_K[0,0]=K[0,0]/ratio
    new_K[1,1]=K[1,1]/ratio

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, DIM, cv2.CV_16SC2)



    # print(map1, map2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img

def padding(img,width,height):
    H = img.shape[0]
    W = img.shape[1]
    top = (height - H) // 2 
    bottom = (height - H) // 2 
    if top + bottom + H < height:
        bottom += 1
    left = (width - W) // 2 
    right = (width - W) // 2 
    if left + right + W < width:
        right += 1
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value = (0,0,0)) 
                             #copyMakeBorder 함수는 이미지를 액자 형태로 만들 때 사용할 수 있습니다. 이미지에 가장자리가 추가
    return img

def color_balance(image): # 색 밸런싱하는 함수
    b, g, r = cv2.split(image)
    B = np.mean(b)
    G = np.mean(g)
    R = np.mean(r)
    K = (R + G + B) / 3
    Kb = K / B
    Kg = K / G
    Kr = K / R
    cv2.addWeighted(b, Kb, 0, 0, 0, b)
    cv2.addWeighted(g, Kg, 0, 0, 0, g)
    cv2.addWeighted(r, Kr, 0, 0, 0, r)
    return cv2.merge([b,g,r])

def luminance_balance(images): # 이미지의 HSV를 통일해주는 함수
    [front,back,left,right] = [cv2.cvtColor(image,cv2.COLOR_BGR2HSV)  
                               for image in images]
                               # -> RGB 색상 이미지를 H(Hue, 색조), S(Saturation, 채도), V(Value, 명도) HSV 이미지로 변형
    hf, sf, vf = cv2.split(front)   # 멀티 채널 Matrix를 여러 개의 싱글 채널 Matrix로 바꿔준다.
    hb, sb, vb = cv2.split(back)    # H,S,V 각각으로 분해된 값이다.
    hl, sl, vl = cv2.split(left)
    hr, sr, vr = cv2.split(right)
    V_f = np.mean(vf) # 주어진 배열의 산술 평균을 반환
    V_b = np.mean(vb)
    V_l = np.mean(vl)
    V_r = np.mean(vr)
    V_mean = (V_f + V_b + V_l +V_r) / 4
    vf = cv2.add(vf,(V_mean - V_f)) # V_mean - V_f = 전체 명도 평균 - FRONT 명도 평균 의 값과 front의 명도를 더함
    vb = cv2.add(vb,(V_mean - V_b)) # 이렇게 더해서 모든 bird Eye View 이미지의 명도 값을 평균적으로 변환
    vl = cv2.add(vl,(V_mean - V_l))
    vr = cv2.add(vr,(V_mean - V_r))
    front = cv2.merge([hf,sf,vf]) # 여러 개의 싱글 채널 Matrix를 멀티 채널 Matrix로 바꿔준다. split의 반대
    back = cv2.merge([hb,sb,vb])
    left = cv2.merge([hl,sl,vl])
    right = cv2.merge([hr,sr,vr])
    images = [front,back,left,right]
    images = [cv2.cvtColor(image,cv2.COLOR_HSV2BGR) for image in images]

    return images



# 1. 4 방향 이미지 read
cap1 = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(2)
cap3 = cv2.VideoCapture(3)
cap4 = cv2.VideoCapture(4)

cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap3.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap3.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap4.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap4.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

while(True):

    ret1, frame1 = cap1.read()    # Read 결과와 frame
    ret2, frame2 = cap2.read()
    ret3, frame3 = cap3.read()
    ret4, frame4 = cap4.read()

# 2. 왜곡 보정
    undistorted_front = undistort(frame1, 1.5)         
    undistorted_back = undistort(frame2, 1.5)        
    undistorted_right = undistort(frame3, 1.5)        
    undistorted_left = undistort_left(frame4, 1.5)   


# 3. 탑뷰 전환 (호모그래피)
    top_front = cv2.warpPerspective(undistorted_front, front_homography, (340, 376)) 
    top_back = cv2.warpPerspective(undistorted_back, back_homography, (340, 376)) 
    top_right = cv2.warpPerspective(undistorted_right, right_homography, (340, 376)) 
    top_left = cv2.warpPerspective(undistorted_left, left_homography, (340, 376)) 


# 4. 이미지 합성
    # masks = [Mask('front'), Mask('back'), Mask('left'), Mask('right')]
    images = [top_front,top_back,top_left,top_right]    
    images = luminance_balance(images)  

    # images = [mask(img) #15
    #             for img, mask in zip(images, self.masks)]
    surround = cv2.add(images[0],images[1]) #이미지를 합침
    surround = cv2.add(surround,images[2])
    surround = cv2.add(surround,images[3])

    surround = color_balance(surround) #16


    cv2.namedWindow('surround', flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.imshow('surround', surround)
    # cv2.imwrite("C:\SSAFY\python\images\surroundView\\surround3.jpg", surround)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()




    surround = bev(front, back, left, right)         

    cv2.namedWindow('surround', flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.imshow('surround', surround)
    cv2.imwrite("C:\SSAFY\python\images\surroundView\\surround3.jpg", surround)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()