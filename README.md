# Dự án Dự Báo Thời Tiết

Dự án này cung cấp hệ thống dự báo thời tiết dựa trên dữ liệu lịch sử. Hệ thống gồm các module thu thập dữ liệu, xử lý dữ liệu, huấn luyện mô hình, dự báo và trực quan hóa kết quả.

## Cấu Trúc Dự Án

- **cities_vn.json** – Chứa thông tin tọa độ và dữ liệu các thành phố dùng trong quá trình thu thập dữ liệu.
- **data/** – Thư mục chứa file CSV:
  - Dữ liệu thời tiết theo giờ (vd: [`Cà Mau_weather.csv`](data/Cà%20Mau_weather.csv)).
  - Dữ liệu thời tiết đã xử lý theo ngày (vd: [`Cà Mau_daily_weather.csv`](data/Cà%20Mau_daily_weather.csv)).
- **modules/**
  - [`data_collection.py`](modules/data_collection.py) – Chứa hàm [`collect_data_for_city`](modules/data_collection.py#L45) để thu thập dữ liệu từ API Meteostat và hàm [`save_data`](modules/data_collection.py#L36) để lưu dữ liệu vào CSV.
  - [`data_processing.py`](modules/data_processing.py) – Xử lý dữ liệu: tổng hợp theo ngày, tính các feature như chênh lệch nhiệt độ, biến động áp suất, tỉ lệ tốc độ gió/áp suất cũng như xử lý outlier qua hàm [`impute_outliers`](modules/data_processing.py#L3).
  - [`model_training.py`](modules/model_training.py) – Cung cấp hàm [`prepare_training_data`](modules/model_training.py#L9) để chuẩn bị dữ liệu huấn luyện dựa trên cửa sổ thời gian, cùng các hàm huấn luyện và đánh giá mô hình.
  - [`prediction.py`](modules/prediction.py) – Chứa hàm [`load_daily_data`](modules/prediction.py#L5) để tải dữ liệu đã xử lý phục vụ dự báo.
  - [`visualization.py`](modules/visualization.py) – Các hàm trực quan như [`plot_forecast`](modules/visualization.py#L7) và [`plot_model_performance`](modules/visualization.py#L41) để hiển thị kết quả dự báo và đánh giá mô hình.
- **main.py** – Điểm vào chính của ứng dụng, cung cấp giao diện dòng lệnh cho các chức năng: thu thập dữ liệu, xử lý dữ liệu, huấn luyện mô hình, dự báo và trực quan hóa.
- **requirements.txt** – Danh sách các gói Python cần thiết.
- **README.md** – File hướng dẫn dự án này.

## Hướng Dẫn Cài Đặt

### Yêu Cầu

- Cài đặt Python (phiên bản 3.8 trở lên).

### Thiết Lập Môi Trường Ảo

1. Tạo môi trường ảo trong thư mục dự án:

   ```sh
   python -m venv env
   ```

2. Kích hoạt môi trường:

- Trên Windows:

    ```sh
    env\Scripts\activate
    ```

- Trên Unix hoặc macOS:

    ```sh
    source env/bin/activate
    ```

### Cài Đặt Các Gói

Trong thư mục dự án, chạy lệnh sau để cài đặt các gói cần thiết:
    ```sh
    pip install -r requirements.txt
    ```

### Cách Sử Dụng

#### Thu Thập Dữ Liệu

-Chạy script data_collection.py để lấy dữ liệu thời tiết từ API Meteostat dựa trên thông tin trong file cities_vn.json:

-Xử Lý Dữ Liệu:
-Chạy script data_processing.py để tổng hợp dữ liệu theo ngày, tính toán các feature bổ sung và xử lý ngoại lệ:

-Huấn Luyện và Đánh Giá Mô Hình:
-Chạy script model_training.py để chuẩn bị dữ liệu huấn luyện, huấn luyện và đánh giá mô hình:

-Dự Báo:
-Sử dụng module prediction.py hoặc chạy giao diện trong main.py để dự báo kết quả:

-Trực Quan Hóa:
-Dùng script visualization.py để hiển thị biểu đồ dự báo, đánh giá hiệu năng mô hình và các biểu đồ trực quan khác:

## Lưu Ý

-Đảm bảo rằng dữ liệu gốc đã được thu thập đúng cách và lưu trữ trong thư mục data trước khi tiến hành xử lý dữ liệu.
-Nếu gặp lỗi thiếu file dữ liệu xử lý theo ngày (như Hà Nội_daily_weather.csv), hãy chạy lại data_processing.py để tạo file cần thiết.

## License

- Dự án được phát hành theo các điều khoản của file LICENSE.

## Cảm Ơn

-Dữ liệu thời tiết được cung cấp thông qua API Meteostat.
-Các biểu đồ trực quan được thực hiện nhờ Matplotlib, Seaborn, và pandas.
