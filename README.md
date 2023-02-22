# 🛒 SVM 시스템

### 삼성전기 기업연계 프로젝트

<details>
<summary>닛산 캐시카이 자동차의 SVM (타사 SVM)</summary>
<div markdown="1">
  
  <img src="https://user-images.githubusercontent.com/19484971/202961156-5bc5ac89-b355-453f-a87a-67fb10fe6bbe.png" width=600>

  > 당시 SVM 시스템 제작에 참고했던 `닛산 캐시카이`의 SVM<br>
  참고에 도움을 주신 ㅊㅎㄱ 컨설턴트님 다시 한 번 감사합니다!

</div>
</details>

<details>
<summary>SVM 시스템 이미지</summary>
<div markdown="1">
  
  <img src="https://user-images.githubusercontent.com/19484971/202616735-6f23192e-6178-40eb-91fc-a91374316e08.png" width=400>

  > 전방 카메라 이미지(왼), 네 카메라를 활용한 SVM 영상(좌)

</div>
</details>

<details>
<summary>SVM 시스템 체험 사진</summary>
<div markdown="1">

  <img src="https://user-images.githubusercontent.com/19484971/220538893-c92173fe-6898-4abb-a3ef-8cdd004a687f.jpg" width=800>

  > SSAFY 동기분이 SVM 시스템 체험 후 제작한 이미지<br>

</div>
</details>

<details>
<summary>SVM 시연 영상</summary>
<div markdown="1">
  
  <img src="./gif/output.gif">

  > 전방 카메라 이미지(왼), 네 카메라를 활용한 SVM 영상(좌)

</div>
</details>

</br>

## 💁‍♀️ 프로젝트 소개

- **프로젝트 기간**

  - 2022.10.11 ~ 2022.11.21 (7주)
  
- 👨‍👧‍👧 팀원
  
  - 6명 (2D SVM 4명, 3D SVM 2명)

- **기획 배경**
  
많은 운전자들이 자동차 주차 시 사각지대로 인해 숙달되기 전까지는 어려움을 겪으며 이는 자동차의 크기가 클수록 정도가 심합니다.

위의 문제점을 해결하기 위해서 차량의 전후좌우 4방향 카메라 영상을 합성하여 SVM(Surround View Monitor) 시스템을 기획하였습니다. 마치 위에서 내려다 보는 듯한 Top View 영상을 제공하여 주차 및 주행 시 사고를 크게 줄일 수 있도록 합니다.

<details>
<summary>예시 이미지</summary>
<div markdown="1">

  <img src="https://user-images.githubusercontent.com/19484971/220541016-2df87ac5-b48a-4d85-bf93-53c9e790b067.png" width = 400>

  <img src="https://user-images.githubusercontent.com/19484971/220541025-d91a4691-6355-4c94-a7b3-ca5f2a4a151b.png" width = 400>

</div>
</details>

- **주요 기능**

  - 화각 180°이상 카메라 이미지 보정 기능 (왜곡보정 및 이미지정렬)
  - Top View 영상 이미지 (2D) 제공
  - 합성 경계 영역 처리
  - 합성 이미지 밝기/색상 보정 기능

</br>

## 🏆 수상내역

<details>
<summary>SSAFY 자율 프로젝트 우수(반 1등)</summary>
<div markdown="1">

  <img src="https://user-images.githubusercontent.com/19484971/220534635-67b61ee2-6485-4918-8076-f09ffc58a781.jpg" width = 600>

  > 본선(지역별) 발표회 1등 상 

</div>
</details>

<details>
<summary>SSAFY 자율 프로젝트 결선 우수(전국 1등)</summary>
<div markdown="1">
  
  <img src="https://user-images.githubusercontent.com/19484971/220531241-4faf2549-6db5-438d-8aa7-c15be156c853.jpg">

  > 결선 발표회 1등 상 수상(왼), 발표회 기념촬영(좌) <br>
  서울에서 진행한 SSAFY 결선(전국) 발표회 사진으로 당시 팀원인 김이랑, 박주현, 임진현이 참가하여 찍은 사진입니다.

</div>
</details>

</br>

## 📒 주요 기술

- Windows 10 Pro
- Python(3.8.13)
- Visual Studio Code (1.70.0)
- openCV (4.6.0)

### 📸 영상처리 기술

- 왜곡보정을 위한 카메라 캘리브레이션(Camera Calibration)
- TopView 변환을 위한 이미지 투영(Imaging Geometry)
- 이미지 투영과 합성을 위한 동차좌표 (Homogeneous coordinates)

<img src="https://user-images.githubusercontent.com/19484971/202954698-00de85b2-040d-45dc-846d-6a6ccb92a298.png" width=400>

> 호모그래피(homography) : 한 평면을 다른 평면에 투영(projection)시켰을 때 투영된 대응점들 사이의 일정한 변환관계

![image](https://user-images.githubusercontent.com/19484971/202955513-674f6fd6-0a0c-4a98-85af-e3f37ec3b0b8.png)
![image](https://user-images.githubusercontent.com/19484971/202955529-fcfe90e4-ef50-498f-9df1-65ce66827a90.png)


- 이미지 합성 경계면 블랜딩 처리 (색상, 명도 보정)

<img src="https://user-images.githubusercontent.com/19484971/202955210-bc560959-7a5b-4fc2-bfb6-efe00a989ec6.png" width=400>

> RGB 행렬 평균값을 활용하여 색상 보정

<img src="https://user-images.githubusercontent.com/19484971/202955228-9185c4dd-9cfe-4ef9-a3d6-46cc6acf6f90.png" width=400>

> HSV 행렬 산술평균 값을 적용하여 명도 조절

- 영상 속도 향상을 위한 멀티 프로세싱

<img src="https://user-images.githubusercontent.com/19484971/202961245-487441ab-5789-491f-8458-bb34e278639a.png" width=400>

<img src="https://user-images.githubusercontent.com/19484971/202960833-23b7c56b-b57d-4c3e-8dd0-81340a9f75e4.png" width=400>

> 5\~7 프레임에서 19\~20 프레임으로 성능 향상

</br>

## 🛠 SVM 장비

<details>
<summary>마트카트</summary>
<div markdown="1">
  
  <img src="https://user-images.githubusercontent.com/19484971/202383691-cbfe1e29-a2dc-4116-a4ad-d3014939b53b.png" width=300>

  - [구입처 & 스팩](https://yestore.kr/shop/item.php?it_id=1426567216)

</div>
</details>

<details>
<summary>어안렌즈 카메라</summary>
<div markdown="1">
  
  <img src="https://user-images.githubusercontent.com/19484971/203259896-15f01710-4c54-47ba-ab3b-9c45517a73e5.png" width=200>

  - [구입처 & 스팩](https://www.coupang.com/vp/products/6595685374?itemId=14901263581&vendorItemId=82139995937&q=%EC%96%B4%EC%95%88%EC%B9%B4%EB%A9%94%EB%9D%BC+usb&itemsCount=36&searchId=2a2f3dea45a94ec59734a168e85c6537&rank=2&isAddedCart=)

</div>
</details>

<details>
<summary>설치 방법</summary>
<div markdown="1">

  <img src="https://user-images.githubusercontent.com/19484971/203239711-6229d112-633e-446c-adbe-fd906a1ca9f8.png" width=400>
  
  > 카메라 설치 위치

  <img src="https://user-images.githubusercontent.com/19484971/203261100-d28591d0-7703-438d-8c89-f2c647dc81f1.png" width=400>
  
  > 카메라 설치 후 실재 모습

  <img src="https://user-images.githubusercontent.com/19484971/203241175-f45d5bea-fa6b-4aa1-bba8-983ff5b42270.png" width=300>

  > 카메라 설치각도

  <img src="https://user-images.githubusercontent.com/19484971/203247200-2268395d-ba5b-4eb9-8d6f-686f36b07608.jpg" width=400>
  
  > 카메라 4대 노트북 연결

</div>
</details>

</br>


## 🤔 추후 개발

<img src="https://user-images.githubusercontent.com/19484971/220555749-7735efbd-a43a-4c1c-9c5b-467c9563934c.gif" width=400>

> 자동차를 중심으로 다양한 시점에서 주변을 확인할 수 있는 3D SVM<br>
Bowl View를 활용하여 제작하였으나, 프레임이 너무 떨어지는 문제가 발생하여 제외

## 👀 더 찾아보기

* [영상처리 이론 정리](https://github.com/ii200400/IT_Skill_Question/tree/master/JobGroup/mobility/cognition/SVM)
* [openGL 실습](https://github.com/ii200400/IT_Skill_Question/tree/master/JobGroup/mobility/cognition/SVM/openGL)
