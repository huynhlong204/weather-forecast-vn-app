import pandas as pd
import os

def process_data(city):
    """
    Đọc dữ liệu thời tiết từ file CSV, tổng hợp theo ngày và lưu vào file mới.
    """
    file_path = os.path.join("data", f"{city}_weather.csv")
    if not os.path.exists(file_path):
        print(f"Không tìm thấy file {file_path}")
        return None
    
    # Đọc dữ liệu dạng hourly
    df = pd.read_csv(file_path, parse_dates=["timestamp"], index_col="timestamp")
    # Tạo cột ngày từ timestamp
    df["date"] = df.index.date
    # Tổng hợp theo ngày: trung bình cho temperature, humidity, wind_speed, pressure và tổng precipitation
    daily = df.groupby("date").agg({
        'temperature': 'mean',
        'humidity': 'mean',
        'wind_speed': 'mean',
        'pressure': 'mean',
        'rain': 'sum'
    }).dropna()
    daily = daily.reset_index()
    
    # Lưu dữ liệu hàng ngày vào file mới
    daily_file = os.path.join("data", f"{city}_daily_weather.csv")
    daily.to_csv(daily_file, index=False)
    print(f"Đã lưu dữ liệu hàng ngày của {city} vào {daily_file}")
    return daily

if __name__ == "__main__":
    city = input("Nhập tên thành phố để xử lý dữ liệu: ")
    process_data(city)
