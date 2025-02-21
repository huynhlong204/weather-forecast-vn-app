import pandas as pd
from meteostat import Point, Hourly
from datetime import datetime, timedelta
import os
import json

def fetch_meteostat_data(lat, lon, days=30):
    """
    Lấy dữ liệu thời tiết từ Meteostat API theo vĩ độ và kinh độ.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    location = Point(lat, lon)
    
    data = Hourly(location, start=start_date, end=end_date)
    data = data.fetch()
    
    if not data.empty:
        data = data.rename(columns={
            'temp': 'temperature',
            'rhum': 'humidity',
            'wspd': 'wind_speed',
            'pres': 'pressure',
            'prcp': 'rain'
        })
        data = data[['temperature', 'humidity', 'wind_speed', 'pressure', 'rain']].dropna()
        data.index.name = 'timestamp'
        return data
    return None

def save_data(city, data):
    """
    Lưu dữ liệu thời tiết vào file CSV trong thư mục data/
    """
    os.makedirs("data", exist_ok=True)
    file_path = f"data/{city}_weather.csv"
    data.to_csv(file_path)
    print(f"Đã lưu dữ liệu thời tiết của {city} vào {file_path}")

def collect_data_for_city(city, days=30):
    """
    Lấy dữ liệu thời tiết cho một thành phố từ file cities_vn.json
    """
    with open("cities_vn.json", "r", encoding="utf-8") as f:
        cities = json.load(f)
    
    if city not in cities:
        print(f"Không tìm thấy thông tin thành phố {city} trong danh sách.")
        return False
    
    coords = cities[city]
    data = fetch_meteostat_data(coords["lat"], coords["lon"], days)
    
    if data is not None:
        save_data(city, data)
        return True
    else:
        print(f"Không lấy được dữ liệu thời tiết cho {city}.")
        return False

if __name__ == "__main__":
    city_name = input("Nhập tên thành phố: ")
    days = int(input("Nhập số ngày lấy dữ liệu: "))
    collect_data_for_city(city_name, days)
