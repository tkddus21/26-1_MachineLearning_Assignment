# 📊 학생 배경 데이터를 활용한 학업 성취도(수학 점수) 예측 모델 연구

> **기계학습(Machine Learning) 텀 프로젝트 최종 결과물**
> * **기관:** 인하대학교 인공지능공학과 (Artificial Intelligence Engineering)
> * **연구자:** 상연 (Sang-yeoni)
> * **선행 연구 배경:** Wenya(2025) 박사학위논문의 교육학적 담론을 기반으로 한 실증적 데이터 분석

---

## 📌 1. 프로젝트 개요 (Project Overview)
본 프로젝트는 학생들의 개인적·사회경제적 배경 데이터(부모 학력, 가계 경제 수준, 사교육 참여 여부 등)를 바탕으로 핵심 학업 성취도 지표인 **수학 점수(`math score`)**를 사전에 예측하는 머신러닝 파이프라인을 구축하는 것을 목표로 합니다. 

학기 초에 학업 부진 위험군 학생을 조기에 선별(Screening)하여 맞춤형 교육 복지 서비스를 제공할 수 있는 **'데이터 기반 조기 경보 시스템(Early Warning System)'**의 기술적 타당성을 검증합니다.

---

## 📂 2. 디렉토리 구조 (Directory Structure)
```text
ml_project/
├── data/
│   └── StudentsPerformance.csv     # Kaggle 오픈 데이터셋
├── src/
│   ├── preprocess.py               # 라벨 인코딩 및 데이터 분할 (6:2:2)
│   ├── train.py                    # 모델 학습, GridSearchCV 튜닝, Failure Mode 분석
│   ├── eda_plots.py                # EDA 시각화 그래프 생성 스크립트
│   └── result_plots.py             # 실험 결과 산점도 및 오차 비교 그래프 생성 스크립트
├── requirements.txt                # 재현성을 위한 가상환경 라이브러리 명세서
└── README.md                       # 본 프로젝트 안내서
```
---
## 3. 개발 환경 및 설치 방법

```
# 1. 저장소 클론
git clone [https://github.com/tkddus21/26-1_MachineLearning_Assignment.git](https://github.com/tkddus21/26-1_MachineLearning_Assignment.git)
cd ml-student-performance-predict

# 2. 가상환경 생성 및 활성화
python3 -m venv venv
source venv/bin/activate

# 3. 의존성 라이브러리 일괄 설치 (과제 가이드라인 충족)
pip install -r requirements.txt
```
---
## 4. 실행파일 설명

train.py => 전처리 파이프라인을 거쳐 Baseline(Linear Regression) 모델과 GridSearchCV 최적화를 거친 Main(Random Forest) 모델의 학습 및 최종 Test 평가를 진행하고, 상위 5개의 실패 모드(Failure Mode) 로그를 출력합니다.

eda_plot.py, result_plots.py =>
데이터 분석 및 결과 시각화 그래프 생성
보고서 및 발표 자료에 사용된 EDA 분포도 그래프와 모델 성능 비교 산점도를 로컬에 이미지 파일로 생성합니다.
## 5. 실험 결과
- Data Split: Train(60%, 600개) | Validation(20%, 200개) | Test(20%, 200개) 격리 분할

### 모델별 최종 성능 정량 비교 (Test Dataset 200개 기준)

안내서 지침에 따라 연속형 회귀 평가 지표인 RMSE와 MAE를 기준으로 평가를 진행했습니다.

---

분석 모델 (Model),최적 하이퍼파라미터 (Hyperparameters),Test RMSE,Test MAE


Linear Regression (Baseline),N/A (선형 파라미터 자동 산출),12.9084,10.1961


Random Forest (Main Model),"max_depth: 5, n_estimators: 100",15.0008,12.0279

---
## 6. 참고문헌 및 출처
[1] Kaggle, "Students Performance in Exams" Open Dataset, Published by spscientist, URL: https://www.kaggle.com/datasets/spscientist/students-performance-in-exams/data (Retrieved: 2026-05-31). 

[2] Pedregosa, F. et al., "Scikit-learn: Machine Learning in Python", Journal of Machine Learning Research, Vol. 12, pp. 2825-2830, 2011. 

[3] Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani, "An Introduction to Statistical Learning: with Applications in R", Springer, 2013. 

[4] Wenya, Yu. (2025). 가정배경, 가정 내 사회적 자본, 사교육, 학습태도 및 학업성취 간의 구조적 관계 [박사학위논문, 경북대학교]. https://www.riss.kr/link?id=T17180600


