# model_utils.py
import xgboost as xgb
import torch
import torch.nn as nn
import numpy as np

# Tải model XGBoost
def load_xgb_model(path=r'C:\Users\nguye\CODE\TimeSeries\BTL\Full_6Thang\XGBoots_LSTM_TFT\XGB_LSTM_TFT_App\best_xgboost_model.json'):
    model = xgb.XGBRegressor()
    model.load_model(path)
    return model

# LSTM đơn giản
class SimpleLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])
        return out

# Tải model LSTM
def load_lstm_model(path=r'C:\Users\nguye\CODE\TimeSeries\BTL\Full_6Thang\XGBoots_LSTM_TFT\XGB_LSTM_TFT_App\best_lstm_model.pth'):
    model = SimpleLSTM()
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Dự đoán với LSTM
def predict_lstm(model, data):  # data: list of float
    x = torch.tensor(data, dtype=torch.float32).view(1, -1, 1)
    with torch.no_grad():
        return model(x).item()
