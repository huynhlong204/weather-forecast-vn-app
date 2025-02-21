import pandas as pd
import matplotlib.pyplot as plt
import os
from modules.prediction import predict_next_7_days, load_daily_data

def plot_forecast(city, window_size=7):
    """
    Hiển thị biểu đồ nhiệt độ lịch sử và dự đoán 7 ngày tiếp theo.
    """
    df = load_daily_data(city)
    if df is None:
        return
    
    preds = predict_next_7_days(city, window_size)
    if preds is None:
        return
    
    # Lấy ngày cuối từ dữ liệu lịch sử
    last_date = pd.to_datetime(df['date']).max()
    # Tạo mảng ngày cho dự đoán: 7 ngày tiếp theo
    pred_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 8)]
    
    plt.figure(figsize=(10, 5))
    plt.plot(pd.to_datetime(df['date']), df['temperature'], label='Nhiệt độ lịch sử')
    plt.plot(pred_dates, preds, label='Dự đoán nhiệt độ', marker='o', linestyle='--')
    plt.xlabel('Ngày')
    plt.ylabel('Nhiệt độ (°C)')
    plt.title(f'Dự đoán nhiệt độ 7 ngày tiếp theo cho {city}')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    city = input("Nhập tên thành phố để hiển thị dự báo: ")
    plot_forecast(city, window_size=7)
