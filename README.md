# **다국어 영수증 OCR**

## Project Overview
### 프로젝트 목표
 - 다국어 (중국어, 일본어, 태국어, 베트남어)로 작성된 영수증 이미지에 대한 OCR task를 수행
 - 글자 검출만을 수행. 즉, 이미지에서 어떤 위치에 글자가 있는지를 예측하는 모델을 제작

### Dataset
- 학습 데이터셋은 언어당 100장, 총 400장의 영수증 이미지 파일 (.jpg)과 해당 이미지에 대한 주석 정보를 포함한 UFO (.json) 파일로 구성되어 있습니다. 각 UFO의 ‘images’키에 해당하는 JSON 값은 각 이미지 파일의 텍스트 내용과 텍스트 좌표를 포함하고 있습니다.
- 총 이미지 수 : 글자가 포함된 JPG 이미지 (학습 총 400장, 테스트 총 120장)

### Project Structure (프로젝트 구조)
```plaintext
code
└─  model.py            # EAST 모델이 정의된 파일입니다.
└─  loss.py             # 학습을 위한 loss function이 정의되어 있는 파일입니다.
└─  train.py            # 모델의 학습 절차가 정의되어 있는 파일입니다.
└─  inference.py        # 모델의 추론 절차가 정의되어 있는 파일입니다.
└─  dataset.py          # 이미지와 글자 영역의 정보 등을 제공하는 데이터셋이 정의되어 있는 파일입니다.
└─  detect.py           # 모델의 추론에 필요한 기타 함수들이 정의되어 있는 파일입니다.
└─  deteval.py          # DetEval 평가를 위한 함수들이 정의되어 있는 파일입니다.
└─  east_dataset.py     # EAST 학습에 필요한 형식의 데이터셋이 정의되어 있는 파일입니다.
└─  requirements.txt    # 패키지 설치를 위한 파일입니다.
└─  pth/                # ImageNet 사전학습 가중치가 들어있는 폴더입니다.
└─  ensemble.py         # nms, wbf 앙상블이 구현되어 있는 파일입니다.
└─  visualization.py    # 결과 시각화를 진행한 이미지 데이터가 저장되는 파일입니다.
└─  data/
    └─ chinese_receipt/
    │  └─ ufo
    │  │    └─ train.json
    │  │    └─ test.json
    │  │    └─ sample_submission.csv
    │  └─ images
    │        └─ train/
    │        └─ test/
    └─ japanses_receipt/
    └─ thai_receipt/
    └─ vietnamese_receipt/
└─  data_prepocess/
    └─ angle_process.ipynb     # 설정한 임계각도 이하인 bbox만 남기는 파일입니다.
    └─ get_new_dataset.ipynb   # 외부 데이터셋을 학습이 가능하게 변환하는 파일입니다.
└─  tools/
    └─ coco_to_ufo.py          # labeling을 위한 포맷 변환 파일입니다. (coco -> ufo)
    └─ to_pickle_no_val.py     # validation set을 포함하지 않은 pkl 파일을 만드는 파일입니다.
    └─ to_pickle_val.py        # validation set을 포함한 pkl 파일을 만드는 파일입니다.
    └─ ufo_to_coco.py          # labeling을 위한 포맷 변환 파일입니다. (ufo -> coco)
    └─ visul.ipynb             # 같은 이미지 데이터셋에 대하여 여러 결과를 한눈에 볼 수 있는 파일입니다.
```

### 협업 tools
- Slack, Notion, Github, Wandb

### GPU
- V100(vram 32GB) 4개

### 평가기준
- 제출된 파일은 7강에서 소개되는 DetEval 방식으로 평가됩니다. DetEval은 이미지 레벨에서 정답 박스가 여러개 존재하고, 예측한 박스가 여러 개가 있을 경우, 박스끼리의 다중 매칭을 허용하여 점수를 주는 평가 방법입니다.
- recall, precision, 과 F1-score 가 기록되고, recall과 precision 의 조화평균인 F1 score이 기준


<br>
  
## Team Introduction
### Members
| 이해강 | 조재만 | 이진우 | 성의인 | 정원정 |
|:--:|:--:|:--:|:--:|:--:|
| <img src="https://github.com/user-attachments/assets/aad9eeae-db0e-41ac-a5ee-d4f12bb9f135" alt="이해강" height="150" width="150"> | <img src="https://github.com/user-attachments/assets/b5d74dd3-d7cf-4697-b8e6-d7047e3f0922" alt="조재만" height="150" width="150"> | <img src="https://github.com/user-attachments/assets/63d1f219-7c86-4591-9183-6f599684a338" alt="이진우" height="150" width="150"> | <img src="https://github.com/user-attachments/assets/604942d4-a6aa-494e-8841-c89f20cef4a6" alt="성의인" height="150" width="150"> | <img src="https://github.com/user-attachments/assets/7274e65e-1a32-4d88-bfa6-c36bac47a2f0" alt="정원정" height="150" width="150"> |
|[Github](https://github.com/lazely)|[Github](https://github.com/PGSammy)|[Github](https://github.com/MUJJINUNGAE)|[Github](https://github.com/EuiInSeong)|[Github](https://github.com/wonjeongjeong)|
|lz0136@naver.com|gaemanssi2@naver.com|dlvy9@naver.com|see8420@naver.com|wj3714@naver.com|
<br>

### Members' Role

| 팀원 | 역할 |
| -- | -- |
| 이해강_T7233 | - 데이터 후처리 가공 <br> - 가설 설정 및 검증  <br> - 시각화 |
| 조재만_T7253 | - data annotation(ICDAR) <br> - 베이스라인 코드 수정 및 최적화 <br> -  가이드라인 제작 <br> - 데이터 labeling 검수 <br> - 데이터 relabeling <br> - 데이터 format 가공 |
| 이진우_T7231 | - EDA <br> - 데이터 relabeling <br> - 결과 시각화 <br> - offline 데이터셋 제작 |
| 성의인_T7166 | - data annotation(SynthText) <br> - 외부 데이터셋 실험 <br> -  데이터 relabeling <br> - 앙상블 구현 <br> - dataset 실험 |
| 정원정_T7272 | - data annotation(AIHub) <br> -  데이터 relabeling <br> - Wandb sweep <br> - online augmentation |

<br>

## Procedure & Techniques

  

| 분류 | 내용 |
| :--: | -- |
|Data|**영수증 이미지 내의 구분선** <br> - 글자만 검출해야하는데 영수증 내에 구분선이 존재하여 글자로 인식하는 경향이 있음 <br> &nbsp;&nbsp;&nbsp;&nbsp;=> 구분선을 포함한 이미지 데이터셋, 구분선 제외한 이미지 데이터셋을 CVAT을 이용해서 각각 제작 후 Train <br> <br>  **Relabeling** <br> - 구분선 포함한 이미지 데이터셋이 존재하면 구분선을 학습하는 것이 아닌 글자로 인식 <br> &nbsp;&nbsp;&nbsp;&nbsp;=> 구분선 및 특수 기호까지 제외한 annotation으로 relabeling 진행 (0.8071 -> 0.8166)<br>  <br> **외부 데이터셋** <br> - clova-ix/cord의 성능이 가장 좋았음 <br> - SROIE <br> - AIHub, ICDAR, SynthText 에는 한계 존재|
|Online Augmentation|**Albumentation** <br> - Normalize <br> - brightness <br> - CLAHE <br> - togray <br> - ISONoise|
|Offline Augmentation|**데이터 가공** <br> - rmbg <br> - Sharpen <br> - denoise|
|Other Methods|**Ensemble** <br> - Non-Maximum Suppression 적용 |
<br> 

<br>

## Results

### 단일 데이터셋

| Dataset | precision | recall | f1 score |
| :--: | :--: | :--: | :--: |
|clova-ix/cord 150epoch| 0.9185| 0.8578| 0.8872|
|clova-ix/cord 300epoch| 0.9211| 0.8531| 0.8858|
|rmbg 300epoch| 0.9067| 0.8485| 0.8766|
|relabeled delete_wrong_bbox| 0.9046| 0.8485| 0.8757|

<br>

### 앙상블

| Emsemble | Calibration | f1 score (public) |
| :--: | :--: | :--: |
|clova_dataset + rmbg + hg800 + (Augmentation)ISOnoise / IoU=0.5| ✔|0.9070|
|clova_dataset + rmbg + hg200 + (Augmentation)ISOnoise / IoU=0.5| |0.9026|
|clova_dataset + rmbg + hg200 + (Augmentation)ISOnoise / IoU=0.6| |0.8999|
|(offline augmentation) origin + shapening + denoising| ✔|0.9112|
|clova_dataset + rmbg + hg800 + (Augmentation)ISOnoise + offline_dataset / IoU=0.2| |0.8913|
<br>

### 최종 과정 및 결과
<img height=80  width=900px alt="image" src="https://github.com/user-attachments/assets/5127b9d7-ba53-4200-89b3-57d27e27605b">
<img height=80  width=900px alt="image" src="https://github.com/user-attachments/assets/2308b48b-f573-4cdb-9872-6308d2ab25f6">
<br>

### 최종 순위
- :trophy: **Public LB : 9th / 24**
<img  alt="image" src="https://github.com/user-attachments/assets/4164ff58-b423-412f-8949-ca3283f0ddfb" height=150  width=900px>

- :trophy: **Private LB : 9th / 24**
<img  alt="image" src="https://github.com/user-attachments/assets/bec59e29-7b4a-4410-9865-2fd0948b74f1" height=150  width=900px>

