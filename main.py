from tabulate import tabulate
from modules import data_collection, data_processing, model_training, visualization, prediction

def main():
    while True:
        print("\n--- Ứng dụng dự báo thời tiết ---")
        print("1. Thu thập dữ liệu thời tiết")
        print("2. Xử lý dữ liệu (tổng hợp theo ngày & feature engineering)")
        print("3. Huấn luyện mô hình và đánh giá (so sánh Baseline vs Enhanced)")
        print("4. Dự đoán 7 ngày tiếp theo (Enhanced)")
        print("5. Hiển thị biểu đồ dự báo (Enhanced)")
        print("6. Hiển thị hiệu năng mô hình Enhanced (scatter & residuals)")
        print("7. So sánh dự đoán (Time Series Baseline vs Enhanced)")
        print("8. Hiển thị tầm quan trọng của các feature (Enhanced)")
        print("9. Hiển thị xu hướng lịch sử với Moving Average")
        print("10. Hiển thị heatmap tương quan các feature")
        print("11. Hiển thị boxplots các feature")
        print("12. Hiển thị violin plots các feature")
        print("13. Thoát")
        choice = input("Chọn chức năng: ")
        
        if choice == "1":
            city = input("Nhập tên thành phố: ")
            days = int(input("Nhập số ngày lấy dữ liệu: "))
            data_collection.collect_data_for_city(city, days)
            
        elif choice == "2":
            city = input("Nhập tên thành phố để xử lý dữ liệu: ")
            data_processing.process_data(city)
            
        elif choice == "3":
            city = input("Nhập tên thành phố để huấn luyện model: ")
            model_training.train_and_evaluate(city, window_size=7)
            
        elif choice == "4":
            city = input("Nhập tên thành phố để dự đoán (Enhanced): ")
            preds = prediction.predict_next_7_days_enhanced(city, window_size=7)
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
                
        elif choice == "5":
            city = input("Nhập tên thành phố để hiển thị dự báo (Enhanced): ")
            visualization.plot_forecast(city, window_size=7)
            
        elif choice == "6":
            city = input("Nhập tên thành phố để hiển thị hiệu năng mô hình (Enhanced): ")
            visualization.plot_model_performance(city, window_size=7)
            
        elif choice == "7":
            city = input("Nhập tên thành phố để so sánh dự đoán (Time Series): ")
            feature = input("Nhập tên feature muốn so sánh (ví dụ: temperature_mean): ")
            visualization.plot_time_series_comparison(city, window_size=7, feature=feature)
            
        elif choice == "8":
            city = input("Nhập tên thành phố để hiển thị xu hướng lịch sử với Moving Average: ")
            visualization.plot_historical_trends(city)
            
        elif choice == "9":
            city = input("Nhập tên thành phố để hiển thị heatmap tương quan các feature: ")
            visualization.plot_correlation_heatmap(city)
            
        elif choice == "10":
            city = input("Nhập tên thành phố để hiển thị boxplots các feature: ")
            visualization.plot_boxplots(city)
            
        elif choice == "11":
            city = input("Nhập tên thành phố để hiển thị violin plots các feature: ")
            visualization.plot_violin_plots(city)
            
        elif choice == "12":
            print("Thoát ứng dụng. Hẹn gặp lại!")
            break
            
        else:
            print("Lựa chọn không hợp lệ. Vui lòng chọn lại.")

if __name__ == "__main__":
    main()
