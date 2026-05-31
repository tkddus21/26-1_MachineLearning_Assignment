import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# 1. 데이터 로드 및 전처리 (학습 코드와 완전히 동일한 파이프라인)
data_path = '/Users/sang-yeoni/Documents/대학교강의자료/3_1/기계학습/ml_project/data/StudentsPerformance.csv'
df = pd.read_csv(data_path)

categorical_cols = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

X = df.drop(columns=['math score', 'reading score', 'writing score'])
y = df['math score']

# 데이터 분할 (6:2:2 구조의 Test 20%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
_, X_test, _, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 2. 모델 학습
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

rf_model = RandomForestRegressor(max_depth=5, n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# 스타일 설정
sns.set_theme(style="whitegrid")
plt.rcParams['font.size'] = 12

# =========================================================================
# [그래프 1] 평가지표(RMSE, MAE) 모델별 정량 비교 바 차트
# =========================================================================
metrics_data = {
    'Model': ['Linear Regression', 'Linear Regression', 'Random Forest', 'Random Forest'],
    'Metric': ['RMSE', 'MAE', 'RMSE', 'MAE'],
    'Score': [12.9084, 10.1961, 15.0008, 12.0279]
}
metrics_df = pd.DataFrame(metrics_data)

plt.figure(figsize=(9, 6))
ax = sns.barplot(x='Metric', y='Score', hue='Model', data=metrics_df, palette='muted')
plt.title('Model Performance Comparison (Lower is Better)', fontsize=14, fontweight='bold')
plt.ylabel('Error Score')
plt.xlabel('Evaluation Metrics')
plt.ylim(0, 18)

# 바 차트 위에 숫자 레이블 표시
for p in ax.patches:
    if p.get_height() > 0:
        ax.annotate(f"{p.get_height():.2f}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 8), 
                    textcoords='offset points', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('model_performance_comparison.png', dpi=300)
plt.close()

# =========================================================================
# [그래프 2] 실제 점수 vs 예측 점수 산점도 및 잔차 분석 회귀선
# =========================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
fig.suptitle('Actual vs Predicted Math Scores', fontsize=16, fontweight='bold')

# 선형 회귀 산점도
ax1.scatter(y_test, lr_preds, alpha=0.6, color='#4c72b0', edgecolors='w', s=50)
ax1.plot([0, 100], [0, 100], color='red', linestyle='--', linewidth=2, label='Perfect Prediction (Y=X)')
ax1.set_title('Linear Regression (Baseline)\nRMSE: 12.91 | MAE: 10.20')
ax1.set_xlabel('Actual Math Score')
ax1.set_ylabel('Predicted Math Score')
ax1.set_xlim(-5, 105)
ax1.set_ylim(-5, 105)
ax1.legend(loc='upper left')

# 랜덤 포레스트 산점도
ax2.scatter(y_test, rf_preds, alpha=0.6, color='#55a868', edgecolors='w', s=50)
ax2.plot([0, 100], [0, 100], color='red', linestyle='--', linewidth=2, label='Perfect Prediction (Y=X)')
ax2.set_title('Random Forest (Main Model)\nRMSE: 15.00 | MAE: 12.03')
ax2.set_xlabel('Actual Math Score')
ax2.set_xlim(-5, 105)
ax2.legend(loc='upper left')

plt.tight_layout()
plt.savefig('actual_vs_predicted_scatters.png', dpi=300)
plt.close()

print("결과 시架화 완료: 'model_performance_comparison.png', 'actual_vs_predicted_scatters.png'가 생성되었습니다.")