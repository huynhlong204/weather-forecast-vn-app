import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from .prediction import predict_next_7_days_enhanced, predict_next_7_days_baseline, load_daily_data

def plot_forecast(city, window_size=7):
    """
    Hiển thị biểu đồ dự báo 7 ngày tiếp theo dựa trên dữ liệu lịch sử và dự đoán của mô hình Enhanced.
    """
    df = load_daily_data(city)
    if df is None:
        return
    preds = predict_next_7_days_enhanced(city, window_size)
    if preds is None:
        return
    
    features = ["temperature_mean", "humidity", "wind_speed", "pressure", "rain"]
    dates_history = pd.to_datetime(df['date'])
    last_date = dates_history.max()
    pred_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 8)]
    
    n_features = len(features)
    fig, axs = plt.subplots(n_features, 1, figsize=(10, 4*n_features), sharex=True)
    if n_features == 1:
        axs = [axs]
    
    for ax, feature in zip(axs, features):
        ax.plot(dates_history, df[feature], label='Historical')
        ax.plot(pred_dates, preds[feature], label='Forecast (Enhanced)', marker='o', linestyle='--')
        ax.set_ylabel(feature)
        ax.legend()
        ax.grid(True)
    
    axs[-1].set_xlabel('Date')
    plt.suptitle(f'7-Day Forecast for {city} (Enhanced)', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_model_performance(city, window_size=7):
    """
    Hiển thị biểu đồ hiệu năng của mô hình Enhanced:
      - Scatter plot Actual vs Predicted cho mỗi feature Enhanced.
      - Histogram phân bố residual cho mỗi feature.
    """
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from .model_training import prepare_training_data
    import joblib
    daily_file = os.path.join("data", f"{city}_daily_weather.csv")
    if not os.path.exists(daily_file):
        print(f"Data file not found: {daily_file}. Please run data_processing.py first.")
        return
    df = pd.read_csv(daily_file)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date').reset_index(drop=True)
    
    enhanced_features = ["temperature_mean", "humidity", "wind_speed", "pressure", "rain", 
                         "temp_diff", "pressure_change", "wind_pressure_ratio"]
    X_all, y_all = prepare_training_data(df, window_size, enhanced_features)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
    
    model_path = os.path.join("models", f"{city}_weather_model_enhanced.pkl")
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}. Please train the model first.")
        return
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    
    features = enhanced_features
    n_features = len(features)
    fig, axs = plt.subplots(n_features, 2, figsize=(12, 4*n_features))
    if n_features == 1:
        axs = [axs]
    for i, feature in enumerate(features):
        axs[i, 0].scatter(y_test[feature], y_pred[:, i], alpha=0.7)
        axs[i, 0].plot([y_test[feature].min(), y_test[feature].max()], [y_test[feature].min(), y_test[feature].max()], 'r--')
        axs[i, 0].set_title(f"Actual vs Predicted: {feature}")
        axs[i, 0].set_xlabel("Actual")
        axs[i, 0].set_ylabel("Predicted")
        residuals = y_test[feature] - y_pred[:, i]
        axs[i, 1].hist(residuals, bins=20, color='skyblue', edgecolor='black')
        axs[i, 1].set_title(f"Residuals: {feature}")
        axs[i, 1].set_xlabel("Residual")
        axs[i, 1].set_ylabel("Frequency")
    plt.tight_layout()
    plt.show()

def plot_time_series_comparison(city, window_size=7, feature="temperature_mean"):
    """
    So sánh dự đoán 7 ngày tiếp theo giữa mô hình Baseline và Enhanced cho một feature cụ thể.
    """
    df = load_daily_data(city)
    if df is None:
        return
    dates_history = pd.to_datetime(df['date'])
    last_date = dates_history.max()
    pred_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 8)]
    preds_baseline = predict_next_7_days_baseline(city, window_size)
    preds_enhanced = predict_next_7_days_enhanced(city, window_size)
    if preds_baseline is None or preds_enhanced is None:
        return
    plt.figure(figsize=(10,6))
    plt.plot(dates_history, df[feature], label="Historical", marker='o')
    plt.plot(pred_dates, preds_baseline[feature], label="Baseline Forecast", marker='o', linestyle='--')
    plt.plot(pred_dates, preds_enhanced[feature], label="Enhanced Forecast", marker='o', linestyle='--')
    plt.title(f"Forecast Comparison for {feature} (Baseline vs Enhanced)")
    plt.xlabel("Date")
    plt.ylabel(feature)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_feature_importances(city):
    """
    Hiển thị biểu đồ tầm quan trọng của các feature trung bình từ mô hình Enhanced.
    """
    import numpy as np
    import joblib
    model_path = os.path.join("models", f"{city}_weather_model_enhanced.pkl")
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}. Please train the model first.")
        return
    model = joblib.load(model_path)
    importances = []
    for estimator in model.estimators_:
        importances.append(estimator.feature_importances_)
    importances = np.array(importances)
    avg_importances = importances.mean(axis=0)
    features = ["temperature_mean", "humidity", "wind_speed", "pressure", "rain", 
                "temp_diff", "pressure_change", "wind_pressure_ratio"]
    plt.figure(figsize=(10,6))
    plt.bar(features, avg_importances, color='skyblue')
    plt.title("Feature Importances (Enhanced Model)")
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Additional visualizations

def plot_historical_trends(city):
    """
    Plot historical trends for all features with moving averages.
    """
    df = load_daily_data(city)
    if df is None:
        return
    df['date'] = pd.to_datetime(df['date'])
    features = ["temperature_mean", "humidity", "wind_speed", "pressure", "rain", 
                "temp_diff", "pressure_change", "wind_pressure_ratio"]
    plt.figure(figsize=(12, 8))
    for feature in features:
        plt.plot(df['date'], df[feature], label=feature)
        plt.plot(df['date'], df[feature].rolling(window=7).mean(), linestyle='--', label=f"{feature} (7-day MA)")
    plt.title(f"Historical Trends for {city}")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(city):
    """
    Plot a correlation heatmap of all features from daily data.
    """
    df = load_daily_data(city)
    if df is None:
        return
    features = ["temperature_mean", "humidity", "wind_speed", "pressure", "rain", 
                "temp_diff", "pressure_change", "wind_pressure_ratio"]
    corr = df[features].corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(f"Correlation Heatmap for {city}")
    plt.tight_layout()
    plt.show()

def plot_boxplots(city):
    """
    Plot boxplots for all features to visualize their distribution.
    """
    df = load_daily_data(city)
    if df is None:
        return
    features = ["temperature_mean", "humidity", "wind_speed", "pressure", "rain", 
                "temp_diff", "pressure_change", "wind_pressure_ratio"]
    plt.figure(figsize=(12, 6))
    df[features].boxplot()
    plt.title(f"Boxplots of Features for {city}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_violin_plots(city):
    """
    Plot violin plots for all features to visualize their distributions.
    """
    df = load_daily_data(city)
    if df is None:
        return
    features = ["temperature_mean", "humidity", "wind_speed", "pressure", "rain", 
                "temp_diff", "pressure_change", "wind_pressure_ratio"]
    df_melted = df.melt(id_vars=['date'], value_vars=features, var_name="Feature", value_name="Value")
    plt.figure(figsize=(12, 6))
    sns.violinplot(x="Feature", y="Value", data=df_melted)
    plt.title(f"Violin Plots of Features for {city}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Chọn chức năng hiển thị:")
    print("1. Dự báo 7 ngày tiếp theo (Enhanced)")
    print("2. Hiển thị hiệu năng mô hình Enhanced (scatter & residuals)")
    print("3. So sánh dự đoán (Time Series Baseline vs Enhanced)")
    print("4. Hiển thị tầm quan trọng của các feature (Enhanced)")
    print("5. Hiển thị xu hướng lịch sử với Moving Average")
    print("6. Hiển thị heatmap tương quan các feature")
    print("7. Hiển thị boxplots các feature")
    print("8. Hiển thị violin plots các feature")
    choice = input("Chọn chức năng (1-8): ")
    city = input("Nhập tên thành phố: ")
    if choice == "1":
        plot_forecast(city, window_size=7)
    elif choice == "2":
        plot_model_performance(city, window_size=7)
    elif choice == "3":
        feature = input("Nhập tên feature muốn so sánh (ví dụ: temperature_mean): ")
        plot_time_series_comparison(city, window_size=7, feature=feature)
    elif choice == "4":
        plot_feature_importances(city)
    elif choice == "5":
        plot_historical_trends(city)
    elif choice == "6":
        plot_correlation_heatmap(city)
    elif choice == "7":
        plot_boxplots(city)
    elif choice == "8":
        plot_violin_plots(city)
    else:
        print("Lựa chọn không hợp lệ!")
