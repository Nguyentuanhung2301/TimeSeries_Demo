
import streamlit as st
import pandas as pd
import numpy as np
from model_utils import load_xgb_model, load_lstm_model, predict_lstm
import matplotlib.pyplot as plt

st.set_page_config(page_title="Churn Prediction Dashboard", layout="wide")
st.title("📊 Dự báo Churn - XGBoost & LSTM")

tab1, tab2, tab3 = st.tabs(["📤 Dự báo từ file CSV", "🧪 Dự báo thủ công", "📘 Giới thiệu mô hình"])

with tab1:
    st.header("📤 Tải lên dữ liệu người dùng")
    uploaded_file = st.file_uploader("Chọn file CSV chứa dữ liệu", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("🔍 Xem trước dữ liệu:", df.head())

        if {"event_time", "event_type", "price", "user_id"}.issubset(df.columns):
            df["event_time"] = pd.to_datetime(df["event_time"])
            df["month"] = df["event_time"].dt.to_period("M").astype(str)
            grouped = df.groupby(["user_id", "month"]).agg(
                event_count=("event_type", "count"),
                total_spent=("price", "sum")
            ).reset_index()

            pivot = grouped.pivot(index="user_id", columns="month", values="event_count").fillna(0)
            st.write("📈 Dữ liệu tổng hợp (event_count):", pivot.head())

            st.write("⚙️ Đang dùng mô hình XGBoost để dự báo churn...")
            model = load_xgb_model()
            X = pivot.values
            preds = model.predict(X)
            pivot["Dự báo churn"] = preds
            st.dataframe(pivot)

with tab2:
    st.header("🧪 Dự báo thủ công")
    model_choice = st.radio("Chọn mô hình", ["XGBoost", "LSTM"])
    default_input = "1000, 950, 900, 920, 910, 880" if model_choice == "XGBoost" else "0.1, 0.12, 0.13, 0.11, 0.15, 0.14, 0.13"
    input_str = st.text_input("📥 Nhập chuỗi dữ liệu:", value=default_input)

    if st.button("Dự báo"):
        try:
            input_data = [float(x.strip()) for x in input_str.split(",")]
            if model_choice == "XGBoost":
                model = load_xgb_model()
                pred = model.predict(np.array(input_data).reshape(1, -1))[0]
            else:
                model = load_lstm_model()
                pred = predict_lstm(model, input_data)
            st.success(f"🔮 Kết quả dự báo ({model_choice}): {pred:.4f}")
        except Exception as e:
            st.error(f"Lỗi: {e}")

with tab3:
    st.header("📘 Giới thiệu mô hình")
    st.markdown("""
    - **XGBoost**: Mô hình cây tăng cường dựa trên dữ liệu đặc trưng từ hành vi người dùng (số event, chi tiêu...)
    - **LSTM**: Dự báo chuỗi thời gian dựa trên hành vi lịch sử
    - Bạn có thể tải dữ liệu hoặc nhập thủ công để thử nghiệm
    """)
