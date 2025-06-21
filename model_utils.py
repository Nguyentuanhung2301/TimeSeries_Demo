
import xgboost as xgb
import torch
import torch.nn as nn
import numpy as np

class SimpleLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])
        return out

def load_xgb_model(path="best_xgboost_model.json"):
    model = xgb.XGBClassifier()
    model.load_model(path)
    return model

def load_lstm_model(path="best_lstm_model.pth"):
    model = SimpleLSTM()
    model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    model.eval()
    return model

def predict_lstm(model, data):
    x = torch.tensor(data, dtype=torch.float32).view(1, -1, 1)
    with torch.no_grad():
        return model(x).item()
