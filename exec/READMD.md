# 포팅메뉴얼

## 카메라 설치

[마트카트](https://yestore.kr/shop/item.php?it_id=1426567216) 전후좌우에 카메라를 설치한다.

<img src="https://user-images.githubusercontent.com/19484971/202383691-cbfe1e29-a2dc-4116-a4ad-d3014939b53b.png" width=500>

카메라를 설치한 위치는 다음과 같다.

--이미지 추가 요망--

웹캠을 컴퓨터에 연결해준다. 즉, C타입을 포함한 USB 포트가 4개 미만이라면 해당 프로젝트는 진행할 수 없다. 단, 카메라 없이 이미지 파일로만 진행하는 것은 가능하다.

-- 웹캠을 모두 연결한 노트북 사진 추가 요망--

## 2D

- 카메라 내부 파라미터 추출
- 카메라 왜곡 보정
- 카메라 homography 적용 및 이미지 합성
- TopView를 통한 2D SVM 제작

### 환경 및 라이브러리

- OS : Windows 10 Pro
- Language : Python(3.8.13), C++
- Visual Studio Code (1.70.0)
- openCV : 4.6.0

python 3.5.3에서도 잘 작동되었다.

### 설치 방법

1. Visual Studio Code, Python 설치

[Visual Studio Code](https://code.visualstudio.com/)는 공식 홈페이지에서 추천하는 버전으로 다운로드 받으면 충분할 것이다.

Python은 [anaconda](https://www.anaconda.com/)를 활용해서 설치해도 되고 [python 공식 홈페이지](https://www.python.org/downloads/)에서 적당한 버전을 설치하면 된다. 

해당 프로젝트는 `Python 3.8.13`을 기준으로 만들었기 때문에 해당 버전을 강력히 추천한다.

2. Visual Studio Code Extension 설치

Visual Studio Code에서 `Extension` 에서 `Python` 설치

![image](https://user-images.githubusercontent.com/19484971/202385665-70638691-85eb-49a1-a2e0-6ff157c90a23.png)

3. openCV 설치

커맨드 창에서 `pip install opencv-python`을 입력하여 openCV를 설치한다.

만약 정상적으로 설치되지 않았다면 `python -m pip install --upgrade pip`을 입력하여 pip를 업데이트한 후 다시 oepncv 설치 명령어를 입력한다.

이후 아래의 코드가 잘 작동되면 설치가 잘 된 것이다.

```
# OpenCV 패키지 임포트
import cv2

# 패키지 설치 위치 확인
print(cv2.__file__)

# 패키지 버전 확인
print(cv2.__version__)
```

4. 카메라 확인

[카메라 테스트용 파일](../SurroundViewMonitor%202D/cameraTest.py)을 실행하여 연결한 카메라가 잘 작동되는지 확인한다. 예시 이미지로만 확인하는 것이라면 생략한다.

보통 인식이 안되는 것은 USB 포트를 뺏다 끼거나 `cv2.VideoCapture(0)`의 숫자를 바꾸어보면서 확인하면 된다. 해당 숫자는 보통 포트를 연결한 순서대로 할당되지만 그렇지 않은 경우도 있기 때문에 꼭 확인하자.

카메라가 잘 실행되고 각 카메라가 원하는 위치의 카메라인지도 확인이 되면 다음 과정을 진행한다. 해당 과정은 카메라를 연결시킬 때마다 시행해주어야 한다.

```
cap1 = cv2.VideoCapture(0)  // 전방 카메라
cap2 = cv2.VideoCapture(1)  // 좌측 카메라
cap3 = cv2.VideoCapture(2)  // 우측 카메라
cap4 = cv2.VideoCapture(3)  // 후방 카메라
```

4. 실행

메인 파일을 실행하여 결과물을 확인하기 전에, 3번에서 확인한 카메라의 숫자를 메인 파일에도 입력해준다. 아래의 코드에서 숫자를 바꾸어주면 된다.

```
cap1 = cv2.VideoCapture(2)
cap2 = cv2.VideoCapture(5)
cap3 = cv2.VideoCapture(4)
cap4 = cv2.VideoCapture(1)
```

만약 예시 이미지로만 확인하는 것이라면 `WORK_PROCESS` 함수 내의 아래의 `cv2.imread`들의 주석을 지우고 `cap2.read()`줄을 주석 처리해주면 된다.

`cv2.imread`에는 원하는 이미지의 경로를 입력해준다. 현재는 프로젝트에서 기본으로 제공하는 예시 이미지로 지정되어있다.

```
# frame = cv2.imread('front.png')
ret1, frame = cap1.read()

# frame = cv2.imread('left.png')
ret2, frame = cap2.read()

# frame = cv2.imread('right.png')
ret2, frame = cap3.read()

# frame = cv2.imread('back.png')
ret3, frame = cap4.read()
```

설정이 완료가 되었다면 [파일](../SurroundViewMonitor%202D/video_capture.py)을 실행하여 결과물을 확인한다. 

실행은 매우 오래걸리기 때문에 길면 15분까지도 기다려주어야 한다. 초기 계산만이 오래 걸리는 것이기 때문에 프로그램 속도와는 연관이 없으니 에러가 생기지 않는 이상 중단시키지 말자. 

카메라의 순서가 맞지 않다면 전,후,좌,우가 다르게 출력될 수 있다. 4개의 사진을 이상하게 이어붙인 듯한 이미지가 보인다면 3번으로 돌아가서 카메라 세팅을 다시 진행하자.

### 결과물

<img src="https://user-images.githubusercontent.com/19484971/202616735-6f23192e-6178-40eb-91fc-a91374316e08.png" width=300>

개발 과정의 이미지들

<img src="https://user-images.githubusercontent.com/19484971/202380608-13c4bd98-325a-44a8-ae4c-6e7c90f9c4c4.png" width=300>

<img src="https://user-images.githubusercontent.com/19484971/202380765-056145a0-20c7-43f3-a2b9-ad1e777a9359.png" width=500>

<img src="https://user-images.githubusercontent.com/19484971/202380881-145c6286-b8bb-433e-b30e-2144a3e03e94.png" width=500>

## 3D

- Bowl 모델을 통한 3D SVM 제작

### 환경 및 라이브러리

- OS : Windows 10 Pro
- Language : C++
- Visual Studio 2017
- openCV : 3.4.15
- openGL
    - glew : 2.1.0 win32
    - glfw : 3.3.8 WIN64
    - glm : 0.9.8.5
    - freeglut-MSVC : 3.0.0-1.mp for MSVC
    - glut : 3.7.6 Win32
- Magick++ : ImageMagick-7.1.0-52-Q16-HDRI

### 설치방법

1. Visual Studio 다운로드

[공식홈페이지](https://learn.microsoft.com/ko-kr/visualstudio/releasenotes/vs2017-relnotes-history)에서 Visual Studio installer 2017버전을 다운로드를 받는다.

<img src="https://user-images.githubusercontent.com/19484971/202615663-33288eda-3ddc-4e32-a820-f9b01cde2923.png" width=500>

현재 2022.11.18일 기준 최신의 2022버전은 해당 프로젝트에서 필요한 SDK나 구성요소를 설치할 수가 없어서 따로 설치를 해주어야 하고 2022버전을 사용한다고 하더라도 플렛폼 도구집합을 `v141`으로 사용해주어야 하기 때문에 2017버전으로 다운로드 해주어야 한다.

2. C++에 대한 필요한 라이브러리를 다운로드

어떤 라이브러리가 필수인지 확실하지는 않지만, 적어도 `C++를 사용한 데스크톱 개발`의 `Windows 8.1 SDK 및 UCRT SDK`, `x86 및 x64용 Visual C++ MFC`는 필수적이다. 

`Windows 10 SDK(10.0.17134.0)`의 경우에는 다른 `Windows 10 SDK`도 작동되는 것 같으므로 하나만 설치하면 될 것이다.

단, 다른 윈도우 버전에서는 확인을 못했으므로 다른 버전이라면 진행이 원활하지 않을 수도 있다. 

<img src="https://user-images.githubusercontent.com/19484971/201578453-b53ba018-f2f5-4bc8-915c-286530e0cb3d.png" width=500>

<img src="https://user-images.githubusercontent.com/19484971/201579769-8f1207eb-501f-4d69-b9a3-a44175ef8b46.png" width=700>

<img src="https://user-images.githubusercontent.com/19484971/201579981-0a828e06-84bb-423c-ac90-7892acaed66c.png" width=400>

2. 솔루션 속성 확인

다른 버전의 Visual Studio로 진행해도 되는 것 같으나 플렛폼 도구집합은 꼭 `v141`로 해주어야 한다. 

또한 `Debug`나 `Release`로 해주는 것은 상관이 없으나, 플렛폼은 꼭 `x64`로 설정해주어야 한다.

<img src="https://user-images.githubusercontent.com/19484971/201580882-5b17af1c-1440-4045-8287-e2ae2fad060b.png" width=500>
<img src="https://user-images.githubusercontent.com/19484971/201581201-55f24c67-1a13-4008-a937-d2bbcae1d0ab.png" width=300>

3. 실행

실행해서 아래와 같은 이미지를 본다면 성공한 것이다!

-- 실행시 이미지 추가 요망--

### 에러

1. `image.cpp`

`image.cpp` 파일이 없다고 에러가 생길 수 있다. 이런 경우 `SurroundViewMonitor\include\Magick++`에 있는 `image.cpp`이 있는 폴더를 참조경로로 넣어주면 된다. 

해당 파일은 [ImageMagick-Windows 깃헙](https://github.com/ImageMagick/ImageMagick-Windows)을 클론하고 파일 내부에 있는 `CloneRepositories.cmd`을 더블클릭하면 자동으로 설치되는 라이브러리 중 `ImageMagick-Windows-main\ImageMagick\Magick++\lib`에서 복사한 것이다.

