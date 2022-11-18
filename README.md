# 주차 보조 시스템 SVM 개발

## 개발환경

- OS : Windows 10 Pro
- Language : Python(3.8.13)
- Code Editor : Visual Studio Code (1.70.0)
- Library
    - openCV (4.6.0)

## 팀원

김기한, 김이랑, 김필재, 박주현, 임영선, 임진현(팀장)

## 프로젝트 기간

2022.10.11 ~ 2022.11.21 (7주)

## 개요

많은 운전자들이 자동차 주차 시 사각지대로 인해 숙달되기 전까지는 어려움을 겪는다. 이는 자동차의 크기가 클수록 정도가 심하다.

위의 문제점을 해결하기 위해서 SVM(Surround View Monitor) 시스템을 개발하여 차량의 전후좌우 4방향 카메라 영상을 합성하여 마치 위에서 내려다 보는 듯한 Top View 영상을 제공하여 주차 및 주행 시 사고를 크게 줄일 수 있도록 한다.

## 이용 대상자

- 차량 이용자

## 주요 기능

- Top View 영상 이미지 (2D) 제공
- 화각 180°이상 카메라 이미지 보정 기능 (왜곡보정 및 이미지정렬)
- 합성 경계 영역 처리
- 합성 이미지 밝기/색상 보정 기능

## 주요 기술

- 왜곡보정을 위한 카메라 캘리브레이션(Camera Calibration)
- TopView 변환을 위한 이미지 투영(Imaging Geometry)
- 이미지 합성을 위한 동차좌표 (Homogeneous coordinates)

## 추후 개발

- 자동차를 중심으로 다양한 시점에서 주변을 확인할 수 있는 Bowl View 영상(3D SVM) 제공

## 결과물

<img src="https://user-images.githubusercontent.com/19484971/202616735-6f23192e-6178-40eb-91fc-a91374316e08.png" width=300>