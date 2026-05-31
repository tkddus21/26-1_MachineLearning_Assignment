import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(data_path):
    df = pd.read_csv(data_path)
    
    # 범주형 변수 처리
    categorical_cols = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
        
    # 수학 점수(math score)를 Target으로 설정
    X = df.drop(columns=['math score', 'reading score', 'writing score'])
    y = df['math score']
    
    # Train / Validation / Test 분할 [cite: 28]
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    return X_train, X_val, X_test, y_train, y_val, y_test