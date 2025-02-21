import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

def prepare_training_data(df, window_size=7, features=["temperature", "humidity", "wind_speed", "pressure", "rain"]):
    """
    Tạo dữ liệu huấn luyện:
    - X: flatten của cửa sổ gồm window_size ngày (mỗi ngày có 5 yếu tố)
    - y: giá trị của ngày kế tiếp cho tất cả các yếu tố.
    """
    X, y = [], []
    for i in range(window_size, len(df)):
        window = df[features].iloc[i-window_size:i].values.flatten()  # kích thước: window_size * len(features)
        target = df[features].iloc[i].values  # giá trị của ngày thứ i
        X.append(window)
        y.append(target)
    return pd.DataFrame(X), pd.DataFrame(y, columns=features)

def train_model(city, window_size=7):
    """
    Huấn luyện mô hình dự báo cho tất cả 5 yếu tố thời tiết dựa trên dữ liệu hàng ngày đã xử lý.
    """
    daily_file = os.path.join("data", f"{city}_daily_weather.csv")
    if not os.path.exists(daily_file):
        print(f"Không tìm thấy dữ liệu hàng ngày cho {city}. Hãy chạy data_processing.py trước.")
        return None
    
    df = pd.read_csv(daily_file)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')
    
    X, y = prepare_training_data(df, window_size=window_size)
    
    # Dùng MultiOutputRegressor để dự đoán nhiều giá trị cùng lúc
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
    model.fit(X, y)
    
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", f"{city}_weather_model.pkl")
    joblib.dump(model, model_path)
    print(f"Model đã được huấn luyện và lưu tại {model_path}")
    return model

if __name__ == "__main__":
    city = input("Nhập tên thành phố để huấn luyện model: ")
    train_model(city, window_size=7)
