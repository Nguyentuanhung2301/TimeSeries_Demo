
import torch.nn as nn

class LSTMChurn(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
