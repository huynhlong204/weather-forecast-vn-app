import pandas as pd
import os
import joblib

def load_daily_data(city):
    """
    Tải dữ liệu hàng ngày đã xử lý.
    """
    daily_file = os.path.join("data", f"{city}_daily_weather.csv")
    if not os.path.exists(daily_file):
        print(f"Không tìm thấy file {daily_file}. Hãy chạy data_processing.py trước.")
        return None
    df = pd.read_csv(daily_file, parse_dates=["date"])
    df = df.sort_values(by='date')
    return df

def predict_next_7_days(city, window_size=7):
    """
    Dự đoán nhiệt độ 7 ngày tiếp theo sử dụng model đã huấn luyện.
    Thuật toán: dùng window_size ngày gần nhất để dự đoán ngày kế tiếp, sau đó cập nhật cửa sổ theo cách lặp.
    """
    # Tải model
    model_path = os.path.join("models", f"{city}_temperature_model.pkl")
    if not os.path.exists(model_path):
        print(f"Không tìm thấy model cho {city}. Hãy huấn luyện model trước.")
        return None
    model = joblib.load(model_path)
    
    # Tải dữ liệu hàng ngày
    df = load_daily_data(city)
    if df is None:
        return None
    
    # Lấy window_size ngày gần nhất
    recent_window = df['temperature'].iloc[-window_size:].tolist()
    
    predictions = []
    for _ in range(7):
        X_input = [recent_window]  # Mảng có hình dạng (1, window_size)
        pred = model.predict(X_input)[0]
        predictions.append(pred)
        # Cập nhật cửa sổ: loại bỏ phần tử đầu, thêm dự đoán mới vào cuối
        recent_window.pop(0)
        recent_window.append(pred)
    
    return predictions

if __name__ == "__main__":
    city = input("Nhập tên thành phố để dự đoán: ")
    preds = predict_next_7_days(city, window_size=7)
    if preds is not None:
        print("Dự đoán nhiệt độ 7 ngày tiếp theo:")
        for i, p in enumerate(preds, start=1):
            print(f"Ngày {i}: {p:.2f} °C")
