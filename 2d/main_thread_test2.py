import cv2
import numpy as np
import argparse
import time
import multiprocessing


parser = argparse.ArgumentParser(description="Generate Surrounding Camera Bird Eye View")
parser.add_argument('-fw', '--FRAME_WIDTH', default=1280, type=int, help='Camera Frame Width')      # 원본 이미지 길이
parser.add_argument('-fh', '--FRAME_HEIGHT', default=720, type=int, help='Camera Frame Height')    # 원본 이미지 높이
parser.add_argument('-bew', '--BEV_WIDTH', default= 680, type=int, help='BEV Frame Width')       # 탑뷰 이미지 길이
parser.add_argument('-beh', '--BEV_HEIGHT', default= 752, type=int, help='BEV Frame Height')     # 탑뷰 이미지 높이
parser.add_argument('-cw', '--CAR_WIDTH', default=133, type=int, help='Car Frame Width')        # 차량 이미지 길이
parser.add_argument('-ch', '--CAR_HEIGHT', default=267, type=int, help='Car Frame Height')      # 차량 이미지 높이
parser.add_argument('-fs', '--FOCAL_SCALE', default=0.65, type=float, help='Camera Undistort Focal Scale')     # 카메라 왜곡되지 않은 초점 스케일
parser.add_argument('-ss', '--SIZE_SCALE', default=1, type=float, help='Camera Undistort Size Scale')       # 카메라 왜곡되지 않은 크기 스케일
parser.add_argument('-blend','--BLEND_FLAG', default=False, type=bool, help='Blend BEV Image (Ture/False)')
parser.add_argument('-balance','--BALANCE_FLAG', default=False, type=bool, help='Balance BEV Image (Ture/False)')
args = parser.parse_args()

FRAME_WIDTH = args.FRAME_WIDTH
FRAME_HEIGHT = args.FRAME_HEIGHT
BEV_WIDTH = args.BEV_WIDTH
BEV_HEIGHT = args.BEV_HEIGHT
CAR_WIDTH = args.CAR_WIDTH
CAR_HEIGHT = args.CAR_HEIGHT
FOCAL_SCALE = args.FOCAL_SCALE
SIZE_SCALE = args.SIZE_SCALE


parser2 = argparse.ArgumentParser(description="Homography from Source to Destination Image")
parser2.add_argument('-bw','--BORAD_WIDTH', default=14, type=int, help='Chess Board Width (corners number)')
parser2.add_argument('-bh','--BORAD_HEIGHT', default=5, type=int, help='Chess Board Height (corners number)')
parser2.add_argument('-size','--SCALED_SIZE', default=10, type=int, help='Scaled Chess Board Square Size (image pixel)')
parser2.add_argument('-subpix_s','--SUBPIX_REGION_SRC', default=3, type=int, help='Corners Subpix Region of img_src')
parser2.add_argument('-subpix_d','--SUBPIX_REGION_DST', default=3, type=int, help='Corners Subpix Region of img_dst')
parser2.add_argument('-store_path', '--STORE_PATH', default='./data/', type=str, help='Path to Store Centerd/Scaled Images')
args2 = parser2.parse_args()


DIM=(1280, 720)
LEFT_K=np.array([[486.43710381577273, 0.0, 643.0021325671074], [0.0, 485.584911786959, 402.9808925210084], [0.0, 0.0, 1.0]])
LEFT_D=np.array([[-0.06338733272909226], [-0.007861033496168955], [0.005073683389947028], [-0.0010639404289377306]])

new_LEFT_K = LEFT_K.copy()
new_LEFT_K[0,0]=LEFT_K[0,0]/1.5
new_LEFT_K[1,1]=LEFT_K[1,1]/1.5
left_map1, left_map2 = cv2.fisheye.initUndistortRectifyMap(LEFT_K, LEFT_D, np.eye(3), new_LEFT_K, DIM, cv2.CV_16SC2)


DIM=(1280, 720)
K=np.array([[455.8515274977241, 0.0, 655.7621645964248], [0.0, 455.08604281075947, 367.3548823943176], [0.0, 0.0, 1.0]])
D=np.array([[-0.02077978156022359], [-0.02434621475644252], [0.009725498728069807], [-0.0018108318059442028]])

## new_K 설정 
new_K = K.copy()
new_K[0,0]=K[0,0]/1.5
new_K[1,1]=K[1,1]/1.5

map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, DIM, cv2.CV_16SC2)


def undistort_left(img, ratio):    
    global left_map1, left_map2 

    undistorted_img = cv2.remap(img, left_map1, left_map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img

def undistort(img, ratio):
    global map1, map2
    
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

class BevGenerator:
    def __init__(self, blend=args.BLEND_FLAG, balance=args.BALANCE_FLAG): #2
        self.init_args()
        self.blend = blend # Bird Eye View 이미지 혼합상태 판단 boolean
        self.balance = balance # Bird Eye View 이미지 균형상태 판단 boolean
        if not self.blend:
            self.masks = [Mask('front'), Mask('back'),  # 10-1
                          Mask('left'), Mask('right')]
        else:
            # print("1111")
            self.masks = [BlendMask('front'), BlendMask('back'), #10-2
                      BlendMask('left'), BlendMask('right')]

    @staticmethod
    def get_args():
        return args

    def init_args(self):
        global FRAME_WIDTH, FRAME_HEIGHT, BEV_WIDTH, BEV_HEIGHT
        global CAR_WIDTH, CAR_HEIGHT, FOCAL_SCALE, SIZE_SCALE
        FRAME_WIDTH = args.FRAME_WIDTH
        FRAME_HEIGHT = args.FRAME_HEIGHT
        BEV_WIDTH = args.BEV_WIDTH
        BEV_HEIGHT = args.BEV_HEIGHT
        CAR_WIDTH = args.CAR_WIDTH
        CAR_HEIGHT = args.CAR_HEIGHT
        FOCAL_SCALE = args.FOCAL_SCALE
        SIZE_SCALE = args.SIZE_SCALE

    def __call__(self, front, back, left, right, car = None):
        images = [front,back,left,right]

        if self.balance:
            images = luminance_balance(images)  #14
        images = [mask(img) #15
                  for img, mask in zip(images, self.masks)]
        surround = cv2.add(images[0],images[1]) #이미지를 합침
        surround = cv2.add(surround,images[2])
        surround = cv2.add(surround,images[3])
        if self.balance:
            surround = color_balance(surround) #16
        if car is not None:
            surround = cv2.add(surround,car)
        return surround

class Mask:
    def __init__(self, name):
        self.mask = self.get_mask(name) #11-1
        
    def get_points(self, name): # Bird Eye View를 위한 point 값 (변환 좌표)
        if name == 'front':
            points = np.array([
                [0, 0],
                [BEV_WIDTH, 0], 
                [(BEV_WIDTH+CAR_WIDTH)/2, (BEV_HEIGHT-CAR_HEIGHT)/2],
                [(BEV_WIDTH-CAR_WIDTH)/2, (BEV_HEIGHT-CAR_HEIGHT)/2]
            ]).astype(np.int32)
        elif name == 'back':
            points = np.array([
                [0, BEV_HEIGHT],
                [BEV_WIDTH, BEV_HEIGHT],
                [(BEV_WIDTH+CAR_WIDTH)/2, (BEV_HEIGHT+CAR_HEIGHT)/2], 
                [(BEV_WIDTH-CAR_WIDTH)/2, (BEV_HEIGHT+CAR_HEIGHT)/2]
            ]).astype(np.int32)
        elif name == 'left':
            points = np.array([
                [0, 0],
                [0, BEV_HEIGHT], 
                [(BEV_WIDTH-CAR_WIDTH)/2, (BEV_HEIGHT+CAR_HEIGHT)/2],
                [(BEV_WIDTH-CAR_WIDTH)/2, (BEV_HEIGHT-CAR_HEIGHT)/2]
            ]).astype(np.int32)
        elif name == 'right':
            points = np.array([
                [BEV_WIDTH, 0],
                [BEV_WIDTH, BEV_HEIGHT], 
                [(BEV_WIDTH+CAR_WIDTH)/2, (BEV_HEIGHT+CAR_HEIGHT)/2], 
                [(BEV_WIDTH+CAR_WIDTH)/2, (BEV_HEIGHT-CAR_HEIGHT)/2]
            ]).astype(np.int32)
        else:
            raise Exception("name should be front/back/left/right")
        return points
    
    def get_mask(self, name):
        mask = np.zeros((BEV_HEIGHT,BEV_WIDTH), dtype=np.uint8) # 마스크 생성, Bird Eye View 높이, 너비만큼 배열 생성
        points = self.get_points(name) # 12-1

        # img = cv2.fillPoly(mask, [points], 255)
        # cv2.namedWindow("raw_frame", flags = cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        # cv2.imshow("raw_frame", img)
        # cv2.waitKey(0)

        return cv2.fillPoly(mask, [points], 255)
        # cv2.fillPoly = 채워진 다각형을 그립니다. pts에 다각형 배열 값을 여러 개 입력할 수도 있습니다. 255 = color
        # fillPoly()에 다각형 좌표 배열을 여러 개 적용한 경우 겹치는 부분이 사라집니다.
    
    def __call__(self, img):
        return cv2.bitwise_and(img, img, mask=self.mask) # mask 영역에서 서로 공통으로 겹치는 부분 출력

class BlendMask:
    def __init__(self,name):
        mf = self.get_mask('front')
        mb = self.get_mask('back')
        ml = self.get_mask('left')
        mr = self.get_mask('right')
        self.get_lines()
        if name == 'front':
            mf = self.get_blend_mask(mf, ml, self.lineFL, self.lineLF)
            mf = self.get_blend_mask(mf, mr, self.lineFR, self.lineRF)
            self.mask = mf
        if name == 'back':
            mb = self.get_blend_mask(mb, ml, self.lineBL, self.lineLB)
            mb = self.get_blend_mask(mb, mr, self.lineBR, self.lineRB)
            self.mask = mb
        if name == 'left':
            ml = self.get_blend_mask(ml, mf, self.lineLF, self.lineFL)
            ml = self.get_blend_mask(ml, mb, self.lineLB, self.lineBL)
            self.mask = ml
        if name == 'right':
            mr = self.get_blend_mask(mr, mf, self.lineRF, self.lineFR)
            mr = self.get_blend_mask(mr, mb, self.lineRB, self.lineBR)
            self.mask = mr
        self.weight = np.repeat(self.mask[:, :, np.newaxis], 3, axis=2) / 255.0
        
        self.weight = self.weight.astype(np.float32)
        
    def get_points(self, name):  # Bird Eye View를 위한 point 값 (변환 좌표)
        if name == 'front':
            points = np.array([
                [0, 0],
                [BEV_WIDTH, 0], 
                [BEV_WIDTH, BEV_HEIGHT/5], 
                [(BEV_WIDTH+CAR_WIDTH)/2, (BEV_HEIGHT-CAR_HEIGHT)/2],
                [(BEV_WIDTH-CAR_WIDTH)/2, (BEV_HEIGHT-CAR_HEIGHT)/2],
                [0, BEV_HEIGHT/5], 
            ]).astype(np.int32)
        elif name == 'back':
            points = np.array([
                [0, BEV_HEIGHT],
                [BEV_WIDTH, BEV_HEIGHT],
                [BEV_WIDTH, BEV_HEIGHT - BEV_HEIGHT/5],
                [(BEV_WIDTH+CAR_WIDTH)/2, (BEV_HEIGHT+CAR_HEIGHT)/2], 
                [(BEV_WIDTH-CAR_WIDTH)/2, (BEV_HEIGHT+CAR_HEIGHT)/2],
                [0, BEV_HEIGHT - BEV_HEIGHT/5],
            ]).astype(np.int32)
        elif name == 'left':
            points = np.array([
                [0, 0],
                [0, BEV_HEIGHT], 
                [BEV_WIDTH/5, BEV_HEIGHT], 
                [(BEV_WIDTH-CAR_WIDTH)/2, (BEV_HEIGHT+CAR_HEIGHT)/2],
                [(BEV_WIDTH-CAR_WIDTH)/2, (BEV_HEIGHT-CAR_HEIGHT)/2],
                [BEV_WIDTH/5, 0]
            ]).astype(np.int32)
        elif name == 'right':
            points = np.array([
                [BEV_WIDTH, 0],
                [BEV_WIDTH, BEV_HEIGHT], 
                [BEV_WIDTH - BEV_WIDTH/5, BEV_HEIGHT],
                [(BEV_WIDTH+CAR_WIDTH)/2, (BEV_HEIGHT+CAR_HEIGHT)/2], 
                [(BEV_WIDTH+CAR_WIDTH)/2, (BEV_HEIGHT-CAR_HEIGHT)/2],
                [BEV_WIDTH - BEV_WIDTH/5, 0]
            ]).astype(np.int32)
        else:
            raise Exception("name should be front/back/left/right")
        return points
    
    def get_mask(self, name): 
        mask = np.zeros((BEV_HEIGHT,BEV_WIDTH), dtype=np.uint8) # 마스크 생성, Bird Eye View 높이, 너비만큼 배열 생성
        points = self.get_points(name)  # 12-2
        return cv2.fillPoly(mask, [points], 255)    # cv2.fillPoly = 채워진 다각형을 그립니다. pts에 다각형 배열 값을 여러 개 입력할 수도 있습니다. 255 = color
        # fillPoly()에 다각형 좌표 배열을 여러 개 적용한 경우 겹치는 부분이 사라집니다.
    
    def get_lines(self):
        self.lineFL = np.array([
                        [0, BEV_HEIGHT/5], 
                        [(BEV_WIDTH-CAR_WIDTH)/2, (BEV_HEIGHT-CAR_HEIGHT)/2],
                    ]).astype(np.int32)
        self.lineFR = np.array([
                        [BEV_WIDTH, BEV_HEIGHT/5], 
                        [(BEV_WIDTH+CAR_WIDTH)/2, (BEV_HEIGHT-CAR_HEIGHT)/2],
                    ]).astype(np.int32)
        self.lineBL = np.array([
                        [0, BEV_HEIGHT - BEV_HEIGHT/5], 
                        [(BEV_WIDTH-CAR_WIDTH)/2, (BEV_HEIGHT+CAR_HEIGHT)/2],
                    ]).astype(np.int32)
        self.lineBR = np.array([
                        [BEV_WIDTH, BEV_HEIGHT - BEV_HEIGHT/5], 
                        [(BEV_WIDTH+CAR_WIDTH)/2, (BEV_HEIGHT+CAR_HEIGHT)/2],
                    ]).astype(np.int32)
        self.lineLF = np.array([
                        [BEV_WIDTH/5, 0],
                        [(BEV_WIDTH-CAR_WIDTH)/2, (BEV_HEIGHT-CAR_HEIGHT)/2]
                    ]).astype(np.int32)
        self.lineLB = np.array([
                        [BEV_WIDTH/5, BEV_HEIGHT],
                        [(BEV_WIDTH-CAR_WIDTH)/2, (BEV_HEIGHT+CAR_HEIGHT)/2]
                    ]).astype(np.int32)
        self.lineRF = np.array([
                        [BEV_WIDTH - BEV_WIDTH/5, 0],
                        [(BEV_WIDTH+CAR_WIDTH)/2, (BEV_HEIGHT-CAR_HEIGHT)/2]
                    ]).astype(np.int32)
        self.lineRB = np.array([
                        [BEV_WIDTH - BEV_WIDTH/5, BEV_HEIGHT],
                        [(BEV_WIDTH+CAR_WIDTH)/2, (BEV_HEIGHT+CAR_HEIGHT)/2]
                    ]).astype(np.int32)
        
    def get_blend_mask(self, maskA, maskB, lineA, lineB): #maskA 값을 합성하는 함수 
        overlap = cv2.bitwise_and(maskA, maskB) # mask 영역에서 서로 공통으로 겹치는 부분 출력
        
        indices = np.where(overlap != 0)
        # print(indices)
    #zip() 함수는 여러 개의 순회 가능한(iterable) 객체를 인자로 받고, 
    # 각 객체가 담고 있는 원소를 터플의 형태로 차례로 접근할 수 있는 반복자(iterator)를 반환합니다. 
    # 설명이 좀 어렵게 들릴 수도 있는데요. 간단한 예제를 보면 이해가 쉬우실 겁니다.
    #>>> numbers = [1, 2, 3]
    #>>> letters = ["A", "B", "C"]
    #>>> for pair in zip(numbers, letters):
    #...     print(pair)
    #...
    #(1, 'A')
    #(2, 'B')
    #(3, 'C')

        for y, x in zip(*indices):
            distA = cv2.pointPolygonTest(np.array(lineA), (x.astype(np.int16), y.astype(np.int16)), True)
            distB = cv2.pointPolygonTest(np.array(lineB), (x.astype(np.int16), y.astype(np.int16)),  True)
            # 이미지에서 해당 Point가 Contour의 어디에 위치해 있는지 확인하는 함수이다.

            # 1. contour : Contour Points들을 인자로 받는다.
            # 2. pt : Contour에 테스트할 Point를 인자로 받는다.
            # 3. measureDist : boolean 데이터 타입. 
            maskA[y, x] = distA**2 / (distA**2 + distB**2 + 1e-6) * 255
        return maskA
    
    def __call__(self, img): 
        return (img * self.weight).astype(np.uint8)   

front_homography = np.array( [[2.5497687517147827, 3.7714976196027767, -1305.9160615726332], [0.030862052170984967, 5.059221608198009, -931.4750390726253], [9.725404703608287e-05, 0.01104129508400122, 1.0]])
back_homography = np.array([[-2.4069730736783495, 2.9205430982962506, 1867.7871790347892], [-0.1060967932993048, 2.054683518354428, 1574.5818360404066], [-0.00019438787613884742, 0.00857226861416168, 1.0]])
right_homography = np.array([[-0.056907775045504295, 1.0672176811325182, 1345.8228403927208], [1.9441218165848122, 2.3632792237771953, -948.6254186141198], [-0.00011958960708869353, 0.006792484660770964, 1.0]])
left_homography = np.array([[0.06600348375424579, 3.8068476399183395, -809.0417825745883], [-2.094846225115616, 2.3159310874593304, 1758.6799688514989], [0.0002774157265438967, 0.006716667751837278, 0.9999999999999999]])


def cam_read(side,d):
    print('cam_read')
    prev_time = 0
    total_frames = 0  
    while True:   
        curr_time = time.time()
        if side == 'front':                   
            ret1, d['front_frame'] = cap1.read()
            # d['front_frame_check'], d['front_frame'] = True, cv2.imread('C:/Users/multicampus/Desktop/S07P31D108/2d/data/extrinsic/front.png')

            d['front_undi'] = cv2.remap(d['front_frame'], map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)   
            d['front_top'] = cv2.warpPerspective(d['front_undi'], front_homography, (680, 752)) 
            d['front_frame_check'] = True
            

        elif side == 'back':       
            ret2, d['back_frame'] = cap2.read()    

            d['back_undi'] = cv2.remap(d['back_frame'], map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)     
            d['back_top'] = cv2.warpPerspective(d['back_undi'], back_homography, (680, 752)) 
            d['back_frame_check'] = True
           
        elif side == 'left':            
            # d['left_frame_check'], d['left_frame'] = cap3.read()
            ret3, d['left_frame'] = True, cv2.imread('C:/Users/multicampus/Desktop/S07P31D108/2d/data/extrinsic/left.png')

            d['left_undi'] = cv2.remap(d['left_frame'], left_map1, left_map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)    
            d['left_top'] = cv2.warpPerspective(d['left_undi'], left_homography, (680, 752)) 
            d['left_frame_check'] = True
                                        
 
        else:        
            # d['right_frame_check'], d['right_frame'] = cap4.read()    
            ret4, d['right_frame'] = True, cv2.imread('C:/Users/multicampus/Desktop/S07P31D108/2d/data/extrinsic/right.png')

            d['right_undi']  = cv2.remap(d['right_frame'], map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)     
            d['right_top'] = cv2.warpPerspective(d['right_undi'], right_homography, (680, 752))
            d['right_frame_check'] = True
           
        total_frames = total_frames + 1
        term = curr_time - prev_time
        fps = 1 / term

        prev_time = curr_time
        fps_string = f'카메라 == {side} + undi + top ===> term = {term:.3f},  FPS = {fps:.2f}'
        print(fps_string)



def undi_top(d):
    
    print('undi_top')
    
    prev_time = 0
    total_frames = 0  
    while True:   
        curr_time = time.time()    
        # print(front_frame)  
        if False not in [d['front_frame_check'], d['back_frame_check'], d['left_frame_check'], d['right_frame_check']]:
                                                
            d['front_undi'] = cv2.remap(d['front_frame'], map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)   
            d['front_top'] = cv2.warpPerspective(d['front_undi'], front_homography, (680, 752))        
                                    
            d['back_undi'] = cv2.remap(d['back_frame'], map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)     
            d['back_top'] = cv2.warpPerspective(d['back_undi'], back_homography, (680, 752)) 
                                    
            d['left_undi'] = cv2.remap(d['left_frame'], left_map1, left_map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)    
            d['left_top'] = cv2.warpPerspective(d['left_undi'], left_homography, (680, 752)) 
                                        
            d['right_undi']  = cv2.remap(d['right_frame'], map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)     
            d['right_top'] = cv2.warpPerspective(d['right_undi'], right_homography, (680, 752)) 

            d['top_check'] = True


            total_frames = total_frames + 1
            term = curr_time - prev_time
            fps = 1 / term

            prev_time = curr_time
            fps_string = f'undi----top ===> term = {term:.3f},  FPS = {fps:.2f}'
            print(fps_string)

def birdeye(bb,d):
    prev_time = 0
    total_frames = 0
    print('birdeye')


    d['mode'] = 'front'
    while True:
        curr_time = time.time()

        if False not in [d['front_frame_check'], d['back_frame_check'], d['left_frame_check'], d['right_frame_check']]:       
            d['surround'] = cv2.resize(bev(d['front_top'],d['back_top'],d['left_top'],d['right_top'], car) ,(445,500) )         
            # d['surround_check'] = True

            if d['mode'] == 'front':
                view = d['front_undi'][0+40:720-60,0+100:1280-100]   #620,1080

            elif d['mode'] == 'back':
                view = d['back_undi'][0+40:720-60,0+100:1280-100]   #620,1080

            elif d['mode'] == 'left':
                view = d['left_undi'][0+40:720-60,0+80:1280-80]   #620,1080

            elif d['mode'] == 'right':
                view = d['right_undi'][0+40:720-60,0+120:1280-120]   #620,1080

            
            view = cv2.resize(view,(870,500)) 
            in_screen = cv2.hconcat([view, d['surround']]) 

            screen[190:190+in_screen.shape[0], 50:50+in_screen.shape[1]] = in_screen            
            
            cv2.putText(screen, d['mode'].upper(), (60, 225), font, 1,(0,255,255), 2, cv2.LINE_AA)
            cv2.imshow('screen', screen)

            key = cv2.waitKey(1)

            if key == ord('q'):
                d['mode'] = 'front'
            elif key == ord('w'):
                d['mode'] = 'back'
            elif key == ord('e'):
                d['mode'] = 'left'
            elif key == ord('r'):
                d['mode'] = 'right'

            elif key == ord('c'):
                cv2.destroyAllWindows()

        
            total_frames = total_frames + 1
            term = curr_time - prev_time
            fps = 1 / term

            prev_time = curr_time
            fps_string = f'합성 ==> term = {term:.3f},  FPS = {fps:.2f}'
            print(fps_string)

def show_bev(bev,d):
    
    # cv2.namedWindow('screen', cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty('screen', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
    
    print('show_bev')
    d['mode'] = 'front' 

    prev_time = 0
    total_frames = 0            

    while True:
        # print(mode)
        curr_time = time.time()
        if d['surround_check'] != False:
           
            if d['mode'] == 'front':
                view = d['front_undi'][0+40:720-60,0+100:1280-100]   #620,1080

            elif d['mode'] == 'back':
                view = d['back_undi'][0+40:720-60,0+100:1280-100]   #620,1080

            elif d['mode'] == 'left':
                view = d['left_undi'][0+40:720-60,0+80:1280-80]   #620,1080

            elif d['mode'] == 'right':
                view = d['right_undi'][0+40:720-60,0+120:1280-120]   #620,1080

            
            view = cv2.resize(view,(870,500)) 
            in_screen = cv2.hconcat([view, d['surround']]) 

            screen[190:190+in_screen.shape[0], 50:50+in_screen.shape[1]] = in_screen            
            
            cv2.putText(screen, d['mode'].upper(), (60, 225), font, 1,(0,255,255), 2, cv2.LINE_AA)
            cv2.imshow('screen', screen)

            key = cv2.waitKey(1)

            if key == ord('q'):
                d['mode'] = 'front'
            elif key == ord('w'):
                d['mode'] = 'back'
            elif key == ord('e'):
                d['mode'] = 'left'
            elif key == ord('r'):
                d['mode'] = 'right'

            elif key == ord('c'):
                cv2.destroyAllWindows()
            
            total_frames = total_frames + 1
            term = curr_time - prev_time
            fps = 1 / term

            prev_time = curr_time
            fps_string = f'show ==> term = {term:.3f},  FPS = {fps:.2f}'
            print(fps_string)
    
cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)
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

args = BevGenerator.get_args()            
args.CAR_WIDTH = 103
args.CAR_HEIGHT = 210             
bev = BevGenerator(blend=True, balance=True)


screen = cv2.imread('C:/Users/multicampus/Desktop/screen.jpg')
screen = cv2.resize(screen,(int(1536),int(860)))
font = cv2.FONT_HERSHEY_DUPLEX 
car = cv2.imread('C:/Users/multicampus/Desktop/porche.png')
car = cv2.resize(car,(213,300))
car = padding(car, BEV_WIDTH, BEV_HEIGHT)


if __name__ == '__main__':

    print('process start')
    manager = multiprocessing.Manager()
    d = manager.dict()

    d['front_frame_check'] = False 
    d['back_frame_check'] = False 
    d['left_frame_check'] = False 
    d['right_frame_check'] = False

    d['top_check'] = False

    d['surround_check'] = False
    
    front_read_top = multiprocessing.Process(target=cam_read,args=('front',d))
    back_read_top = multiprocessing.Process(target=cam_read,args=('back',d))
    left_read_top = multiprocessing.Process(target=cam_read,args=('left',d))
    right_read_top = multiprocessing.Process(target=cam_read,args=('right',d))

    # undi_tops = multiprocessing.Process(target=undi_top,args=(d,))
    
    make_bev = multiprocessing.Process(target=birdeye,args=('bev',d))
    # show = multiprocessing.Process(target=show_bev,args=('bev',d))

    front_read_top.start()
    back_read_top.start()
    left_read_top.start()
    right_read_top.start()

    # undi_tops.start()

    make_bev.start()
    # show.start()
    
    processes = [front_read_top,back_read_top,left_read_top,right_read_top,make_bev]
    for process in processes:
        process.join()




