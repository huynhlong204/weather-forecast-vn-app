import pandas as pd
import matplotlib.pyplot as plt
import os
from modules.prediction import predict_next_7_days, load_daily_data

def plot_forecast(city, window_size=7):
    """
    Hiển thị biểu đồ trực quan hóa dữ liệu lịch sử (hàng ngày) và dự đoán 7 ngày tiếp theo
    cho tất cả các yếu tố: temperature, humidity, wind_speed, pressure, rain.
    """
    df = load_daily_data(city)
    if df is None:
        return
    
    preds = predict_next_7_days(city, window_size)
    if preds is None:
        return
    
    features = ["temperature", "humidity", "wind_speed", "pressure", "rain"]
    
    # Lấy ngày từ dữ liệu lịch sử và tạo mảng ngày dự đoán cho 7 ngày tiếp theo
    dates_history = pd.to_datetime(df['date'])
    last_date = dates_history.max()
    pred_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 8)]
    
    # Số lượng subplot theo số yếu tố
    n_features = len(features)
    fig, axs = plt.subplots(n_features, 1, figsize=(10, 4*n_features), sharex=True)
    
    if n_features == 1:
        axs = [axs]
    
    for ax, feature in zip(axs, features):
        ax.plot(dates_history, df[feature], label='Lịch sử')
        ax.plot(pred_dates, preds[feature], label='Dự đoán', marker='o', linestyle='--')
        ax.set_ylabel(feature)
        ax.legend()
        ax.grid(True)
    
    axs[-1].set_xlabel('Ngày')
    plt.suptitle(f'Dự đoán 7 ngày tiếp theo cho {city}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    city = input("Nhập tên thành phố để hiển thị dự báo: ")
    plot_forecast(city, window_size=7)
