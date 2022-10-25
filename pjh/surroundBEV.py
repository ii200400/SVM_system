import cv2
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description="Generate Surrounding Camera Bird Eye View")
parser.add_argument('-fw', '--FRAME_WIDTH', default=1280, type=int, help='Camera Frame Width')
parser.add_argument('-fh', '--FRAME_HEIGHT', default=1024, type=int, help='Camera Frame Height')
parser.add_argument('-bw', '--BEV_WIDTH', default=1000, type=int, help='BEV Frame Width')
parser.add_argument('-bh', '--BEV_HEIGHT', default=1000, type=int, help='BEV Frame Height')
parser.add_argument('-cw', '--CAR_WIDTH', default=250, type=int, help='Car Frame Width')
parser.add_argument('-ch', '--CAR_HEIGHT', default=400, type=int, help='Car Frame Height')
parser.add_argument('-fs', '--FOCAL_SCALE', default=1, type=float, help='Camera Undistort Focal Scale')
parser.add_argument('-ss', '--SIZE_SCALE', default=2, type=float, help='Camera Undistort Size Scale')
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

class Camera:
    def __init__(self, name):   #4
        self.camera_mat = np.load(os.path.dirname(__file__) + '/data/{}/camera_{}_K.npy'.format(name,name)) # npy 파일에서 K값 배열 받아오기
        # 카메라 매트릭스 K
        self.dist_coeff = np.load(os.path.dirname(__file__) + '/data/{}/camera_{}_D.npy'.format(name,name)) # npy 파일에서 D값 배열 받아오기
        # 거리 계수 D
        self.homography = np.load(os.path.dirname(__file__) + '/data/{}/camera_{}_H.npy'.format(name,name)) # npy 파일에서 H값 배열 받아오기
        # 호모그래피 H, 호모그래피란?
        self.camera_mat_dst = self.get_camera_mat_dst() #5
        self.undistort_maps = self.get_undistort_maps() #6
        self.bev_maps = self.get_bev_maps() #7
        
    def get_camera_mat_dst(self):
        camera_mat_dst = self.camera_mat.copy()
        camera_mat_dst[0][0] *= FOCAL_SCALE # 카메라 왜곡되지 않은 초점 스케일 , default=1, type=float
        camera_mat_dst[1][1] *= FOCAL_SCALE
        camera_mat_dst[0][2] = FRAME_WIDTH / 2 * SIZE_SCALE # Camera Frame Width, default=1280, type=int
        camera_mat_dst[1][2] = FRAME_HEIGHT / 2 * SIZE_SCALE # Camera Frame Height, default=1024, type=int
        # SIZE_SCALE = 카메라 왜곡되지 않은 크기 스케일, default=2, type=float
        return camera_mat_dst
    
    def get_undistort_maps(self):
        undistort_maps = cv2.fisheye.initUndistortRectifyMap(
                    self.camera_mat, self.dist_coeff, 
                    np.eye(3, 3), self.camera_mat_dst,  # P----	새로운 카메라 고유 매트릭스(3x3) 또는 새로운 프로젝션 매트릭스(3x4)
                    (int(FRAME_WIDTH * SIZE_SCALE), int(FRAME_HEIGHT * SIZE_SCALE)), cv2.CV_16SC2)
        return undistort_maps
    
    def get_bev_maps(self):
        map1 = self.warp_homography(self.undistort_maps[0]) #8
        map2 = self.warp_homography(self.undistort_maps[1])
        return (map1, map2)
    
    def undistort(self, img):   ################################################## 얘 왜 사용안함?
        return cv2.remap(img, *self.undistort_maps, interpolation = cv2.INTER_LINEAR) #img를 *self.undistort_maps으로 리매핑하는 함수
        
    def warp_homography(self, img): 
        return cv2.warpPerspective(img, self.homography, (BEV_WIDTH,BEV_HEIGHT)) #9
        #원근 변환 함수, 원근 맵 행렬(self.homography)를 (BEV_WIDTH,BEV_HEIGHT) //Bird Eye View = BEV 에 해당하는 출력 이미지 크기로 img를 변환하여 반환
        
    def raw2bev(self, img):
        return cv2.remap(img, *self.bev_maps, interpolation = cv2.INTER_LINEAR) #img를 *self.bev_maps으로 리매핑하는 함수

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
        return cv2.fillPoly(mask, [points], 255)
        # cv2.fillPoly = 채워진 다각형을 그립니다. pts에 다각형 배열 값을 여러 개 입력할 수도 있습니다. 255 = color
        # fillPoly()에 다각형 좌표 배열을 여러 개 적용한 경우 겹치는 부분이 사라집니다.
    
    def __call__(self, img):
        return cv2.bitwise_and(img, img, mask=self.mask) # mask 영역에서 서로 공통으로 겹치는 부분 출력

class BlendMask:
    def __init__(self,name):
        mf = self.get_mask('front') #11-2
        mb = self.get_mask('back')
        ml = self.get_mask('left')
        mr = self.get_mask('right')
        self.get_lines()    #12-2-1
        if name == 'front':
            mf = self.get_blend_mask(mf, ml, self.lineFL, self.lineLF) #12-2-2
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
            distA = cv2.pointPolygonTest(np.array(lineA), (x, y), True)
            distB = cv2.pointPolygonTest(np.array(lineB), (x, y), True)
            # 이미지에서 해당 Point가 Contour의 어디에 위치해 있는지 확인하는 함수이다.

            # 1. contour : Contour Points들을 인자로 받는다.
            # 2. pt : Contour에 테스트할 Point를 인자로 받는다.
            # 3. measureDist : boolean 데이터 타입. 
            maskA[y, x] = distA**2 / (distA**2 + distB**2 + 1e-6) * 255
        return maskA
    
    def __call__(self, img):
        return (img * self.weight).astype(np.uint8)    
    
class BevGenerator:
    def __init__(self, blend=args.BLEND_FLAG, balance=args.BALANCE_FLAG): #2
        self.init_args()
        self.cameras = [Camera('front'), Camera('back'),    #3
                        Camera('left'), Camera('right')]
        self.blend = blend # Bird Eye View 이미지 혼합상태 판단 boolean
        self.balance = balance # Bird Eye View 이미지 균형상태 판단 boolean
        if not self.blend:
            self.masks = [Mask('front'), Mask('back'),  # 10-1
                          Mask('left'), Mask('right')]
        else:
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
        images = [mask(camera.raw2bev(img)) #15
                  for img, mask, camera in zip(images, self.masks, self.cameras)]
        surround = cv2.add(images[0],images[1]) #이미지를 합침
        surround = cv2.add(surround,images[2])
        surround = cv2.add(surround,images[3])
        if self.balance:
            surround = color_balance(surround) #16
        if car is not None:
            surround = cv2.add(surround,car)
        return surround

def main():
    front = cv2.imread('./data/front/front.jpg')
    back = cv2.imread('./data/back/back.jpg')
    left = cv2.imread('./data/left/left.jpg')
    right = cv2.imread('./data/right/right.jpg')
    car = cv2.imread('./data/car.jpg')
    car = padding(car, BEV_WIDTH, BEV_HEIGHT) #차 이미지에 가장자리 추가
    
    bev = BevGenerator() #1
    surround = bev(front,back,left,right,car)   #13
    
    cv2.namedWindow('surround', flags = cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.imshow('surround', surround)
    cv2.imwrite('./surround.jpg', surround)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()