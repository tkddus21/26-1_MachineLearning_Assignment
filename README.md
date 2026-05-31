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
