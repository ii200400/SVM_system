class BevGenerator:
    def __init__(self, blend=args.BLEND_FLAG, balance=args.BALANCE_FLAG): #2
        self.init_args()
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

