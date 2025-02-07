from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
from sklearn.linear_model import LinearRegression
import mplfinance as mpf

# 🔹 تعريف تطبيق Flask
app = Flask(__name__)

def calculate_rsi(df, period=14):
    delta = df["سعر الإغلاق"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    
    return df

def analyze_trend(df):
    df["نسبة التغير"] = ((df["سعر الإغلاق"] - df["سعر الفتح"]) / df["سعر الفتح"]) * 100
    df["الاتجاه"] = df["نسبة التغير"].apply(lambda x: "صعودي" if x > 0 else ("محايد" if abs(x) < 0.2 else "هبوطي"))
    return df

def predict_next_period(df):
    df["index"] = range(len(df))
    X = df[["index"]].values
    y = df["سعر الإغلاق"].values
    
    model = LinearRegression()
    model.fit(X, y)

    next_index = np.array([[len(df)]])
    predicted_close = round(model.predict(next_index)[0], 4)
    
    predicted_high = round(predicted_close + np.std(df["أعلى سعر"] - df["سعر الإغلاق"]), 4)
    predicted_low = round(predicted_close - np.std(df["سعر الإغلاق"] - df["أقل سعر"]), 4)

    return predicted_close, predicted_high, predicted_low

def plot_candlestick_chart(df, predicted_close, predicted_high, predicted_low):
    df["التاريخ"] = pd.date_range(start="2025-01-01", periods=len(df), freq="H")
    df.set_index("التاريخ", inplace=True)

    df_mpf = df.rename(columns={"سعر الفتح": "Open", "أعلى سعر": "High", "أقل سعر": "Low", "سعر الإغلاق": "Close"})

    # 🔹 التأكد من تطابق عدد الأعمدة عند إضافة الصف الجديد
    next_date = df.index[-1] + pd.Timedelta(hours=1)
    new_row = pd.DataFrame({
        "Open": [predicted_close],
        "High": [predicted_high],
        "Low": [predicted_low],
        "Close": [predicted_close]
    }, index=[next_date])

    df_mpf = pd.concat([df_mpf, new_row])  # إضافة الصف الجديد بطريقة صحيحة

    fig, ax = plt.subplots(figsize=(8, 5))
    mpf.plot(df_mpf, type='candle', style='charles', ax=ax)

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return plot_url


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        periods = int(request.form['periods'])
        time_duration = request.form['time_duration']
        time_unit = request.form['time_unit']
        full_time_duration = f"{time_duration} {time_unit}"
        data = []
        for i in range(periods):
            open_price = round(float(request.form[f'open_price_{i}']), 4)
            high_price = round(float(request.form[f'high_price_{i}']), 4)
            low_price = round(float(request.form[f'low_price_{i}']), 4)
            close_price = round(float(request.form[f'close_price_{i}']), 4)
            data.append([i+1, open_price, high_price, low_price, close_price])
        
        df = pd.DataFrame(data, columns=["الفترة", "سعر الفتح", "أعلى سعر", "أقل سعر", "سعر الإغلاق"])
        df = analyze_trend(df)
        df = calculate_rsi(df)
        predicted_close, predicted_high, predicted_low = predict_next_period(df)
        plot_url = plot_candlestick_chart(df, predicted_close, predicted_high, predicted_low)
        
        return render_template('index.html', tables=[df.to_html(classes='data')], plot_url=plot_url, 
                               predicted_close=predicted_close, predicted_high=predicted_high, predicted_low=predicted_low)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
