from modules import data_collection, data_processing, model_training, prediction, visualization

def main():
    while True:
        print("\n--- Ứng dụng dự báo thời tiết ---")
        print("1. Thu thập dữ liệu thời tiết")
        print("2. Xử lý dữ liệu (tổng hợp theo ngày)")
        print("3. Huấn luyện mô hình")
        print("4. Dự đoán 7 ngày tiếp theo")
        print("5. Hiển thị biểu đồ dự báo")
        print("6. Thoát")
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
            model_training.train_model(city, window_size=7)
        elif choice == "4":
            city = input("Nhập tên thành phố để dự đoán: ")
            preds = prediction.predict_next_7_days(city, window_size=7)
            if preds is not None:
                print("Dự đoán nhiệt độ 7 ngày tiếp theo:")
                for i, p in enumerate(preds, start=1):
                    print(f"Ngày {i}: {p:.2f} °C")
        elif choice == "5":
            city = input("Nhập tên thành phố để hiển thị dự báo: ")
            visualization.plot_forecast(city, window_size=7)
        elif choice == "6":
            break
        else:
            print("Lựa chọn không hợp lệ. Vui lòng chọn lại.")

if __name__ == "__main__":
    main()
