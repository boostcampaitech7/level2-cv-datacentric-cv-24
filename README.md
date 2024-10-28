# **다국어 영수증 OCR**

## Project Overview
### 프로젝트 목표
 - 다국어 (중국어, 일본어, 태국어, 베트남어)로 작성된 영수증 이미지에 대한 OCR task를 수행
 - 글자 검출만을 수행. 즉, 이미지에서 어떤 위치에 글자가 있는지를 예측하는 모델을 제작

### Dataset
- 학습 데이터셋은 언어당 100장, 총 400장의 영수증 이미지 파일 (.jpg)과 해당 이미지에 대한 주석 정보를 포함한 UFO (.json) 파일로 구성되어 있습니다. 각 UFO의 ‘images’키에 해당하는 JSON 값은 각 이미지 파일의 텍스트 내용과 텍스트 좌표를 포함하고 있습니다.
- 총 이미지 수 : 글자가 포함된 JPG 이미지 (학습 총 400장, 테스트 총 120장)

### Project Structure (프로젝트 구조)
<!--
```plaintext
project/
├── public/
│   ├── index.html           # HTML 템플릿 파일
│   └── favicon.ico          # 아이콘 파일
├── src/
│   ├── assets/              # 이미지, 폰트 등 정적 파일
│   ├── components/          # 재사용 가능한 UI 컴포넌트
│   ├── hooks/               # 커스텀 훅 모음
│   ├── pages/               # 각 페이지별 컴포넌트
│   ├── App.js               # 메인 애플리케이션 컴포넌트
│   ├── index.js             # 엔트리 포인트 파일
│   ├── index.css            # 전역 css 파일
│   ├── firebaseConfig.js    # firebase 인스턴스 초기화 파일
│   package-lock.json    # 정확한 종속성 버전이 기록된 파일로, 일관된 빌드를 보장
│   package.json         # 프로젝트 종속성 및 스크립트 정의
├── .gitignore               # Git 무시 파일 목록
└── README.md                # 프로젝트 개요 및 사용법
```
-->

### 협업 tools
- Slack, Notion, Github, Wandb

### GPU
- V100(vram 32GB) 4개

### 평가기준
- 제출된 파일은 7강에서 소개되는 DetEval 방식으로 평가됩니다. DetEval은 이미지 레벨에서 정답 박스가 여러개 존재하고, 예측한 박스가 여러개가 있을 경우, 박스끼리의 다중 매칭을 허용하여 점수를 주는 평가방법입니다.
- recall, precision, 과 F1-score 가 기록되고, recall과 precision 의 조화평균인 F1 score이 기준


<br>
  
## Team Introduction
### Members
| 이해강 | 조재만 | 이진우 | 성의인 | 정원정 |
|:--:|:--:|:--:|:--:|:--:|
| <img src="https://github.com/user-attachments/assets/aad9eeae-db0e-41ac-a5ee-d4f12bb9f135" alt="이해강" height="150" width="150"> | <img src="https://github.com/user-attachments/assets/b5d74dd3-d7cf-4697-b8e6-d7047e3f0922" alt="조재만" height="150" width="150"> | <img src="https://github.com/user-attachments/assets/998386df-549b-4069-8a2e-07e3ebe4f9a1" alt="이진우" height="150" width="150"> | <img src="https://github.com/user-attachments/assets/604942d4-a6aa-494e-8841-c89f20cef4a6" alt="성의인" height="150" width="150"> | <img src="https://github.com/user-attachments/assets/7274e65e-1a32-4d88-bfa6-c36bac47a2f0" alt="정원정" height="150" width="150"> |
|[Github](https://github.com/lazely)|[Github](https://github.com/PGSammy)|[Github](https://github.com/MUJJINUNGAE)|[Github](https://github.com/EuiInSeong)|[Github](https://github.com/wonjeongjeong)|
|lz0136@naver.com|gaemanssi2@naver.com|a01041206201@gmail.com|see8420@naver.com|wj3714@naver.com|

### Members' Role

| 팀원 | 역할 |
| -- | -- |
| 이해강_T7233 | - 베이스라인 실험 <br> -  <br> -  <br> -  |
| 조재만_T7253 | - data annotation(ICDAR) <br> - <br> - <br> - |
| 이진우_T7231 | - EDA <br> -  <br> - <br> - |
| 성의인_T7166 | - data annotation(SynthText) <br> -  <br> -  <br> -  |
| 정원정_T7272 | - data annotation(AIHub) <br> - <br> -  <br> -  |

<br>

## Procedure & Techniques

  
<!--
| 분류 | 내용 |
| :--: | -- |
|Data|**Stratified Group K-fold** <br> - 하나의 이미지가 하나의 class에 할당되는 것이 아닌 여러 개의 object(class)를 포함 <br> &nbsp;&nbsp;&nbsp;&nbsp;=> object들의 class별 분포가 최대한 유사하도록 각각 5개의 Train/Valid set(8:2로 분할)을 구성 <br> <br>  **Augmentation** <br> - 각 모델에 기본적인 데이터 증강으로 Horizontal Flip과 Vertical Flip을 적용 <br> - 그 외에도 Rotate, Sharpen, Emboss 등 다양한 augmentation 사용  <br> - 다양한 augmentation을 적용할수록 더 높은 mAP 점수를 보임 <br> <br> **Label Correction** <br> - train dataset의 Paper와 General Trash의 경계가 애매모호하다는 것을 확인 <br> &nbsp;&nbsp;&nbsp;&nbsp;=> 라벨링 기준을 정하여 Correction을 한 결과, mAP50 점수가 상승되었다. (0.5371->0.5420)
|Model|**Cascade-RCNN** <br> - Backbone : Swin-L <br> - Neck : FPN <br> - Head : Cascade-RCNN <br> <br> **ATSS** <br> - Backbone : Swin-L <br> - Neck : FPN <br> - Head : ATSS + Dyhead <br> <br> **Deformable DETR** <br> - Backbone : Swin-L <br> - Neck : Channel Mapper <br> - Head : Deformable DETR Head
|HyperParameters|**Cascade-RCNN** <br> - Batch Size : 32 <br> - Class Loss : Cross Entropy <br> - BoundingBox Loss : Smooth-L1 <br> - Learning Rate : 0.0001 <br> - Optimizer : AdamW <br> - Epochs : 13 <br> <br> **ATSS** <br> - Batch Size : 32 <br> - Class Loss : Focal Loss <br> - BoundingBox Loss : GioU Loss <br> - Learning Rate : 0.00005 <br> - Optimizer : AdamW <br> - Epochs : 18 <br> <br> **DETR** <br> - Batch Size : 32 <br> - Class Loss : Focal Loss <br> - BoundingBox Loss : L1-Loss <br> - Learning Rate : 0.0002 <br> - Optimizer : AdamW <br> - Epochs : 21
|Other Methods|**Ensemble** <br> - Weighted Boxes Fusion <br> - Confidence score calibration 적용 <br> <br>  **Pseudo Labeling** <br> - 주어진 Train dataset 뿐만 아니라 label이 없는 Test dataset까지 학습에 이용해서 모델 성능을 최대한 향상시키기 위함 <br> - ATSS 1epoch 적용 (Public mAP : 0.7157 -> 0.7185)
-->
<br>

## Results

### 단일모델
<!--
| Method | Backbone | mAP50 | mAP75 | mAP50(LB) |
| :--: | :--: | :--: | :--: | :--: |
|Faster RCNN| ResNet101| 0.4845| 0.313 |0.4683|
|DetectoRS| ResNext101| 0.514 |0.385 |0.4801|
|TridentNet |Trident + ResNet101| 0.5341| 0.4311| 0.5428|
|**Cascade RCNN**| Swin-L |0.633| 0.539| 0.6257|
|**Deformable DETR**| Swin-L| 0.621 |0.533| 0.6373|
|**ATSS**| Swin-L| 0.689| 0.596| 0.6741|
-->
<br>

<!--### 앙상블

| Emsemble | Calibration | mAP50(LB) |
| :--: | :--: | :--: |
|ATSS (5Fold), Deformable DETR (5Fold), Swin-L + Cascade (5Fold)| |0.7054|
|ATSS (5Fold), Deformable DETR (5Fold), Swin-L + Cascade (5Fold)| ✔|0.7116|
|ATSS + Pseudo (5Fold), Deformable DETR (5Fold), Swin-L + Cascade (5Fold)| ✔|0.7185|
-->
### 최종 과정 및 결과
<!--<img  src='https://i.esdrop.com/d/f/JnUd4EePHi/RtXESQ1EMi.png'  height=530  width=900px></img>-->

### 최종 순위
<!--- 🥈 **Public LB : 2nd / 19**
<img  src='https://i.esdrop.com/d/f/JnUd4EePHi/pRxm1J5V4K.png'  height=200  width=900px></img>
- 🥈 **Private LB : 2nd / 19**
<img  src='https://i.esdrop.com/d/f/JnUd4EePHi/lH6U2wutr6.png'  height=200  width=900px></img>-->
