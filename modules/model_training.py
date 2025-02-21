import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestRegressor

def prepare_training_data(df, feature='temperature', window_size=7):
    """
    Tạo dữ liệu huấn luyện với cửa sổ thời gian (window_size) cho biến 'feature'.
    Ví dụ: dùng 7 ngày trước để dự đoán ngày kế tiếp.
    """
    X, y = [], []
    for i in range(window_size, len(df)):
        X.append(df[feature].iloc[i-window_size:i].values)
        y.append(df[feature].iloc[i])
    return pd.DataFrame(X), pd.Series(y)

def train_model(city, window_size=7):
    """
    Huấn luyện model dự báo nhiệt độ cho thành phố dựa trên dữ liệu hàng ngày.
    """
    daily_file = os.path.join("data", f"{city}_daily_weather.csv")
    if not os.path.exists(daily_file):
        print(f"Không tìm thấy dữ liệu hàng ngày cho {city}. Hãy chạy data_processing.py trước.")
        return None
    
    df = pd.read_csv(daily_file)
    # Chuyển cột 'date' sang kiểu datetime và sắp xếp theo ngày
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')
    
    X, y = prepare_training_data(df, feature='temperature', window_size=window_size)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Lưu model đã huấn luyện vào thư mục models
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", f"{city}_temperature_model.pkl")
    joblib.dump(model, model_path)
    print(f"Model đã được huấn luyện và lưu tại {model_path}")
    return model

if __name__ == "__main__":
    city = input("Nhập tên thành phố để huấn luyện model: ")
    train_model(city, window_size=7)
