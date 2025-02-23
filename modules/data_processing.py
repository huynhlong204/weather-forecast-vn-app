import pandas as pd
import os

def impute_outliers(df, columns):
    """
    Thay thế các giá trị ngoại lai (outlier) trong các cột số bằng giá trị trung vị của cột đó,
    sử dụng phương pháp IQR.
    """
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        median_val = df[col].median()
        # Thay thế các giá trị vượt quá khoảng giới hạn bằng median
        df[col] = df[col].apply(lambda x: median_val if (x < lower_bound or x > upper_bound) else x)
    return df

def process_data(city):
    """
    Đọc dữ liệu thời tiết từ file CSV (dữ liệu theo giờ), tổng hợp theo ngày, xử lý outlier
    bằng imputation và tạo thêm các feature mới.
    """
    file_path = os.path.join("data", f"{city}_weather.csv")
    if not os.path.exists(file_path):
        print(f"Không tìm thấy file {file_path}")
        return None
    
    # Đọc dữ liệu dạng hourly
    df = pd.read_csv(file_path, parse_dates=["timestamp"], index_col="timestamp")
    
    # Tạo cột ngày từ timestamp
    df["date"] = df.index.date
    
    # Tổng hợp theo ngày:
    # Với temperature: tính trung bình, max, min
    daily = df.groupby("date").agg({
        'temperature': ['mean', 'max', 'min'],
        'humidity': 'mean',
        'wind_speed': 'mean',
        'wind_direction': 'mean',  # Dùng trung bình (cho demo; thực tế có thể xử lý riêng)
        'pressure': 'mean',
        'rain': 'sum'
    })
    daily.columns = ['temperature_mean', 'temperature_max', 'temperature_min', 'humidity', 'wind_speed', 'wind_direction', 'pressure', 'rain']
    daily = daily.dropna().reset_index()
    
    # Xử lý outlier bằng imputation cho các cột số quan trọng
    numeric_cols = ['temperature_mean', 'temperature_max', 'temperature_min', 'humidity', 'wind_speed', 'pressure', 'rain']
    daily = impute_outliers(daily, numeric_cols)
    
    # Tạo các feature mới
    daily["temp_diff"] = daily["temperature_max"] - daily["temperature_min"]            # Chênh lệch nhiệt độ ngày/đêm
    daily["pressure_change"] = daily["pressure"].diff().fillna(0)                       # Biến động áp suất giữa các ngày
    daily["wind_pressure_ratio"] = daily["wind_speed"] / daily["pressure"]              # Tỷ lệ giữa tốc độ gió và áp suất
    
    daily_file = os.path.join("data", f"{city}_daily_weather.csv")
    daily.to_csv(daily_file, index=False)
    print(f"Đã lưu dữ liệu hàng ngày của {city} vào {daily_file}")
    return daily

if __name__ == "__main__":
    city = input("Nhập tên thành phố để xử lý dữ liệu: ")
    process_data(city)
