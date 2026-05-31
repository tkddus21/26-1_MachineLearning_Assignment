import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from preprocess import load_and_preprocess_data

def evaluate_model(name, y_true, y_pred):
    # 안내서 요구사항: 최소 2개 이상의 평가지표 측정 (RMSE, MAE) [cite: 27]
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"[{name} 결과] RMSE: {rmse:.4f} | MAE: {mae:.4f}")
    return rmse, mae

def main():
    # 데이터 경로 지정
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_data('../data/StudentsPerformance.csv')
    
    # 1. Baseline Model 설정 (단순 선형 회귀) [cite: 26]
    baseline_model = LinearRegression()
    baseline_model.fit(X_train, y_train)
    val_preds_base = baseline_model.predict(X_val)
    evaluate_model("Baseline (Linear Regression)", y_val, val_preds_base)
    
    # 2. Main Model & 하이퍼파라미터 튜닝 [cite: 25, 27]
    main_model = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10, None]
    }
    
    grid_search = GridSearchCV(estimator=main_model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"\n최적 하이퍼파라미터: {grid_search.best_params_}")
    
    # 3. 최종 Test 데이터 평가 [cite: 28]
    test_preds = best_model.predict(X_test)
    evaluate_model("Final Main Model (RandomForest)", y_test, test_preds)
    
    # 4. Failure Mode 분석 (가장 크게 틀린 데이터 추출) [cite: 32, 33]
    errors = np.abs(y_test - test_preds)
    worst_indices = np.argsort(errors)[-5:] 
    
    print("\n--- [Failure Mode Analysis] 모델이 가장 크게 틀린 케이스 5개 ---")
    for idx in worst_indices:
        print(f"실제 점수: {y_test.iloc[idx]} | 예측 점수: {test_preds[idx]:.1f} | 오차: {errors.iloc[idx]:.1f}")

if __name__ == "__main__":
    main()