# app.py
import streamlit as st
from model_utils import load_xgb_model, load_lstm_model, predict_lstm
import numpy as np

st.title("Dự báo Churn - XGBoost & LSTM")

input_data = st.text_input("Nhập chuỗi dữ liệu cách nhau bởi dấu phẩy (vd: 123,456,789)", "")

if input_data:
    try:
        data = [float(x.strip()) for x in input_data.split(',')]
        
        # XGB
        xgb_model = load_xgb_model()
        xgb_pred = xgb_model.predict(np.array(data).reshape(1, -1))[0]

        # LSTM
        lstm_model = load_lstm_model()
        lstm_pred = predict_lstm(lstm_model, data)

        st.subheader("Kết quả dự báo:")
        st.write(f"🔹 XGBoost: {xgb_pred:.4f}")
        st.write(f"🔹 LSTM: {lstm_pred:.4f}")
    except Exception as e:
        st.error(f"Lỗi xử lý dữ liệu: {e}")
