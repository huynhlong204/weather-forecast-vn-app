import pandas as pd
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import joblib

def prepare_training_data(df, window_size, features):
    """
    Tạo dữ liệu huấn luyện theo kiểu cửa sổ thời gian:
        - X: Flatten cửa sổ gồm window_size ngày (mỗi ngày gồm các feature)
        - y: Giá trị của ngày kế tiếp cho các feature đó.
    """
    X, y = [], []
    for i in range(window_size, len(df)):
        window = df[features].iloc[i-window_size:i].values.flatten()
        target = df[features].iloc[i].values
        X.append(window)
        y.append(target)
    return pd.DataFrame(X), pd.DataFrame(y, columns=features)

def train_and_evaluate(city, window_size=7):
    """
    Huấn luyện mô hình dự báo với 2 bộ feature:
        - Baseline: sử dụng 5 feature gốc [temperature_mean, humidity, wind_speed, pressure, rain]
        - Enhanced: thêm các feature mới [temp_diff, pressure_change, wind_pressure_ratio]
    So sánh MAE & RMSE của 2 mô hình và lưu lại cả 2 mô hình.
    """
    daily_file = os.path.join("data", f"{city}_daily_weather.csv")
    if not os.path.exists(daily_file):
        print(f"Không tìm thấy dữ liệu hàng ngày cho {city}. Hãy chạy data_processing.py trước.")
        return None
    
    df = pd.read_csv(daily_file)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date').reset_index(drop=True)
    
    baseline_features = ["temperature_mean", "humidity", "wind_speed", "pressure", "rain"]
    enhanced_features = baseline_features + ["temp_diff", "pressure_change", "wind_pressure_ratio"]
    
    # Baseline model training
    X_base, y_base = prepare_training_data(df, window_size, baseline_features)
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_base, y_base, test_size=0.2, random_state=42)
    
    model_base = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
    model_base.fit(X_train_b, y_train_b)
    y_pred_b = model_base.predict(X_test_b)
    
    mae_base = mean_absolute_error(y_test_b, y_pred_b)
    rmse_base = np.sqrt(mean_squared_error(y_test_b, y_pred_b))
    
    print("----- Mô hình Baseline -----")
    print(f"MAE: {mae_base:.2f}, RMSE: {rmse_base:.2f}")
    
    # Enhanced model training
    X_enh, y_enh = prepare_training_data(df, window_size, enhanced_features)
    X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(X_enh, y_enh, test_size=0.2, random_state=42)
    
    model_enh = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
    model_enh.fit(X_train_e, y_train_e)
    y_pred_e = model_enh.predict(X_test_e)
    
    mae_enh = mean_absolute_error(y_test_e, y_pred_e)
    rmse_enh = np.sqrt(mean_squared_error(y_test_e, y_pred_e))
    
    print("----- Mô hình Enhanced -----")
    print(f"MAE: {mae_enh:.2f}, RMSE: {rmse_enh:.2f}")
    
    improvement_mae = ((mae_base - mae_enh) / mae_base) * 100 if mae_base else 0
    improvement_rmse = ((rmse_base - rmse_enh) / rmse_base) * 100 if rmse_base else 0
    
    print("----- So sánh cải thiện -----")
    print(f"Cải thiện MAE: {improvement_mae:.2f}%")
    print(f"Cải thiện RMSE: {improvement_rmse:.2f}%")
    
    os.makedirs("models", exist_ok=True)
    baseline_model_path = os.path.join("models", f"{city}_weather_model_baseline.pkl")
    joblib.dump(model_base, baseline_model_path)
    print(f"Model Baseline đã được lưu tại {baseline_model_path}")
    
    enhanced_model_path = os.path.join("models", f"{city}_weather_model_enhanced.pkl")
    joblib.dump(model_enh, enhanced_model_path)
    print(f"Model Enhanced đã được lưu tại {enhanced_model_path}")
    
    return model_enh

if __name__ == "__main__":
    city = input("Nhập tên thành phố để huấn luyện model: ")
    train_and_evaluate(city, window_size=7)
