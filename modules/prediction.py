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
    df = df.sort_values(by="date")
    return df

def predict_next_7_days(city, window_size=7):
    """
    Dự đoán 7 ngày tiếp theo cho các yếu tố thời tiết: temperature, humidity, wind_speed, pressure, rain.
    Dùng mô hình đã huấn luyện và phương pháp dự đoán theo kiểu lặp.
    """
    model_path = os.path.join("models", f"{city}_weather_model.pkl")
    if not os.path.exists(model_path):
        print(f"Không tìm thấy model cho {city}. Hãy huấn luyện model trước.")
        return None
    model = joblib.load(model_path)
    
    df = load_daily_data(city)
    if df is None:
        return None
    df = df.sort_values(by="date")
    features = ["temperature", "humidity", "wind_speed", "pressure", "rain"]
    
    # Lấy cửa sổ window_size ngày cuối cùng, flatten thành 1 danh sách
    recent_window = df[features].iloc[-window_size:].values.flatten().tolist()
    
    # Lưu dự đoán cho mỗi yếu tố theo kiểu dictionary
    predictions = {feature: [] for feature in features}
    
    for _ in range(7):
        X_input = [recent_window]  # hình dạng: (1, window_size*len(features))
        pred = model.predict(X_input)[0]  # mảng 5 giá trị theo thứ tự của features
        for i, feature in enumerate(features):
            predictions[feature].append(pred[i])
        # Cập nhật cửa sổ: loại bỏ dữ liệu của ngày cũ nhất (5 giá trị) và thêm giá trị dự đoán mới
        recent_window = recent_window[5:] + list(pred)
    
    return predictions

if __name__ == "__main__":
    city = input("Nhập tên thành phố để dự đoán: ")
    preds = predict_next_7_days(city, window_size=7)
    if preds is not None:
        # Định nghĩa các icon cho từng yếu tố
        icons = {
            "temperature": "🌡️",
            "humidity": "💧",
            "wind_speed": "💨",
            "pressure": "🔵",
            "rain": "🌧️"
        }
        
        # Tạo bảng dữ liệu với các cột: Icon, Yếu tố, Dự đoán 7 ngày (dạng chuỗi)
        table = []
        for feature, values in preds.items():
            values_str = ", ".join([f"{float(v):.2f}" for v in values])
            table.append([icons.get(feature, ""), feature.capitalize(), values_str])
            
        headers = ["Icon", "Yếu tố", "Dự đoán 7 ngày"]
        print("\nDự đoán 7 ngày tiếp theo:")
        print(tabulate(table, headers=headers, tablefmt="fancy_grid"))