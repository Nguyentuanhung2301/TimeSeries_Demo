
import streamlit as st
import pandas as pd
import torch
import numpy as np
from xgboost import XGBClassifier
from torch import nn

# Load mô hình
@st.cache_resource
def load_xgb_model():
    model = XGBClassifier()
    model.load_model("models/best_xgboost_model.json")
    return model

@st.cache_resource
def load_lstm_model():
    class LSTMChurn(nn.Module):
        def __init__(self, input_size, hidden_size, n_layers=2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True, dropout=0.3)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.fc(out[:, -1, :])
            return out

    model = LSTMChurn(input_size=4, hidden_size=128)
    model.load_state_dict(torch.load("models/best_lstm_model.pth", map_location="cpu"))
    model.eval()
    return model

st.set_page_config(page_title="Dự báo Churn", layout="wide")
st.title("📊 Dự báo Churn - XGBoost & LSTM")

tabs = st.tabs(["📁 Dự báo từ file CSV", "🧪 Dự báo thủ công", "ℹ️ Giới thiệu mô hình"])

# Tab 1 - Upload file CSV
with tabs[0]:
    st.subheader("📤 Tải lên dữ liệu người dùng")
    uploaded_file = st.file_uploader("Chọn file CSV chứa dữ liệu", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())

        features = ['event_count', 'total_spent', 'purchase_count', 'n_categories']
        if not all(f in df.columns for f in features):
            st.error("❌ File CSV phải chứa các cột: " + ", ".join(features))
        else:
            model_choice = st.radio("Chọn mô hình dự báo", ["XGBoost", "LSTM"])
            X = df[features].values

            if model_choice == "XGBoost":
                model = load_xgb_model()
                preds = model.predict(X)
            else:
                model = load_lstm_model()
                X_seq = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    output = model(X_seq)
                    preds = (torch.sigmoid(output).numpy() > 0.5).astype(int).flatten()

            df['Dự đoán Churn'] = preds
            st.success("✅ Dự đoán hoàn tất!")
            st.dataframe(df)

# Tab 2 - Dự báo thủ công
with tabs[1]:
    st.sidebar.header("📊 Dự báo thủ công")
model_type = st.sidebar.radio("Chọn mô hình", ["XGBoost", "LSTM"])

st.subheader("🧪 Nhập dữ liệu đặc trưng")

event_count = st.number_input("event_count", min_value=0.0, step=1.0)
total_spent = st.number_input("total_spent", min_value=0.0, step=1.0)
purchase_count = st.number_input("purchase_count", min_value=0.0, step=1.0)
n_categories = st.number_input("n_categories", min_value=0.0, step=1.0)

if st.button("Dự báo"):
    input_data = np.array([[event_count, total_spent, purchase_count, n_categories]])

    if model_type == "XGBoost":
        model = load_xgb_model()
        pred = model.predict(input_data)[0]
        st.success(f"🔮 Dự đoán XGBoost: {'Churn' if pred == 1 else 'Không Churn'}")

    else: model_type == "LSTM"
          model = load_lstm_model()
          model.eval()
          with torch.no_grad():
            tensor_input = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
            output = model(tensor_input)
            prob = torch.sigmoid(output).item()
            st.success(f"🔮 Dự đoán LSTM: {'Churn' if prob > 0.5 else 'Không Churn'} (Xác suất: {prob:.2f})")

# Tab 3 - Giới thiệu mô hình
with tabs[2]:
    st.markdown("""
## 🔍 Giới thiệu
Ứng dụng này sử dụng 2 mô hình học máy:
- **XGBoost**: Áp dụng boosting cho bài toán phân loại nhị phân (Churn/Không).
- **LSTM**: Áp dụng học sâu với chuỗi thời gian ngắn để dự đoán churn.

### 📌 Đặc trưng đầu vào:
- `event_count`
- `total_spent`
- `purchase_count`
- `n_categories`

Bạn có thể:
- Tải lên file `.csv` có các cột trên để dự đoán hàng loạt.
- Hoặc nhập tay thủ công 4 giá trị để thử nghiệm mô hình.
""")
