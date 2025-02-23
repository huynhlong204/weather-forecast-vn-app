import pandas as pd
import os
import joblib
from tabulate import tabulate

def load_daily_data(city):
    """
    Tải dữ liệu hàng ngày đã được xử lý.
    """
    daily_file = os.path.join("data", f"{city}_daily_weather.csv")
    if not os.path.exists(daily_file):
        print(f"Không tìm thấy file {daily_file}. Hãy chạy data_processing.py trước.")
        return None
    df = pd.read_csv(daily_file, parse_dates=["date"])
    df = df.sort_values(by="date").reset_index(drop=True)
    return df

def predict_next_7_days_enhanced(city, window_size=7):
    """
    Dự đoán 7 ngày tiếp theo sử dụng mô hình Enhanced.
    """
    model_path = os.path.join("models", f"{city}_weather_model_enhanced.pkl")
    if not os.path.exists(model_path):
        print(f"Không tìm thấy model Enhanced cho {city}. Hãy huấn luyện model trước.")
        return None
    model = joblib.load(model_path)
    
    df = load_daily_data(city)
    if df is None:
        return None
    
    features = ["temperature_mean", "humidity", "wind_speed", "pressure", "rain", 
                "temp_diff", "pressure_change", "wind_pressure_ratio"]
    recent_window = df[features].iloc[-window_size:].values.flatten().tolist()
    predictions = {feature: [] for feature in features}
    
    for _ in range(7):
        X_input = [recent_window]
        pred = model.predict(X_input)[0]
        for i, feature in enumerate(features):
            predictions[feature].append(pred[i])
        recent_window = recent_window[len(features):] + list(pred)
    
    return predictions

def predict_next_7_days_baseline(city, window_size=7):
    """
    Dự đoán 7 ngày tiếp theo sử dụng mô hình Baseline.
    """
    model_path = os.path.join("models", f"{city}_weather_model_baseline.pkl")
    if not os.path.exists(model_path):
        print(f"Không tìm thấy model Baseline cho {city}. Hãy huấn luyện model trước.")
        return None
    model = joblib.load(model_path)
    
    df = load_daily_data(city)
    if df is None:
        return None
    
    features = ["temperature_mean", "humidity", "wind_speed", "pressure", "rain"]
    recent_window = df[features].iloc[-window_size:].values.flatten().tolist()
    predictions = {feature: [] for feature in features}
    
    for _ in range(7):
        X_input = [recent_window]
        pred = model.predict(X_input)[0]
        for i, feature in enumerate(features):
            predictions[feature].append(pred[i])
        recent_window = recent_window[len(features):] + list(pred)
    
    return predictions

if __name__ == "__main__":
    city = input("Nhập tên thành phố để dự đoán (Enhanced): ")
    preds = predict_next_7_days_enhanced(city, window_size=7)
    if preds is not None:
        icons = {
            "temperature_mean": "🌡️",
            "humidity": "💧",
            "wind_speed": "💨",
            "pressure": "🔵",
            "rain": "🌧️",
            "temp_diff": "📈",
            "pressure_change": "⚖️",
            "wind_pressure_ratio": "🔄"
        }
        table = []
        for feature, values in preds.items():
            values_str = ", ".join([f"{v:.2f}" for v in values])
            table.append([icons.get(feature, ""), feature, values_str])
        headers = ["Icon", "Yếu tố", "Dự đoán 7 ngày (Enhanced)"]
        print("\nDự đoán 7 ngày tiếp theo (Enhanced):")
        print(tabulate(table, headers=headers, tablefmt="fancy_grid"))
