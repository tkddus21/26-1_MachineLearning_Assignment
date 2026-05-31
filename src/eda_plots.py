import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 로드
df = pd.read_csv('../data/StudentsPerformance.csv')

# 그래프 스타일 및 한글/영문 폰트 가독성 설정
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['font.size'] = 11

# =========================================================================
# [그래프 1] 성별에 따른 과목별 성적 역전 현상 (Boxplot)
# =========================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
fig.suptitle('Academic Performance Distribution by Gender', fontsize=16, fontweight='bold')

sns.boxplot(ax=axes[0], x='gender', y='math score', data=df, palette='Set2')
axes[0].set_title('Math Scores (Male Dominant)')
axes[0].set_xlabel('Gender')
axes[0].set_ylabel('Score')

sns.boxplot(ax=axes[1], x='gender', y='reading score', data=df, palette='Set2')
axes[1].set_title('Reading Scores (Female Dominant)')
axes[1].set_xlabel('Gender')
axes[1].set_ylabel('')

sns.boxplot(ax=axes[2], x='gender', y='writing score', data=df, palette='Set2')
axes[2].set_title('Writing Scores (Female Dominant)')
axes[2].set_xlabel('Gender')
axes[2].set_ylabel('')

plt.tight_layout()
plt.savefig('eda_gender_comparison.png', dpi=300)
plt.close()

# =========================================================================
# [그래프 2] 경제적 배경(급식) 및 시험 준비 여부에 따른 수학 성적 밀도 (Violinplot)
# =========================================================================
plt.figure(figsize=(12, 6))
sns.violinplot(x='lunch', y='math score', hue='test preparation course', 
               data=df, split=True, inner='quart', palette='muted')

plt.title('Math Score Density by Lunch Type and Test Preparation', fontsize=14, fontweight='bold')
plt.xlabel('Lunch Type (Socioeconomic Status Indicator)')
plt.ylabel('Math Score')
plt.legend(title='Test Prep Course')

plt.tight_layout()
plt.savefig('eda_lunch_vs_prep.png', dpi=300)
plt.close()

print("EDA 시각화 이미지 2종 생성 완료: 'eda_gender_comparison.png', 'eda_lunch_vs_prep.png'")