from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import torch
import torch.nn as nn
import joblib

class BiGRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, dropout_prob=0.3):
        super(BiGRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size * 2, 64)

    def forward(self, x):
        _, h = self.gru(x)
        h = h.permute(1, 0, 2).reshape(x.size(0), -1)
        h = self.dropout(h)
        return self.fc(h)

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, 32)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class FusionModel(nn.Module):
    def __init__(self, input_dim=102, num_classes=6):
        super(FusionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, gru_out, cnn_out, lgb_out):
        x = torch.cat([gru_out, cnn_out, lgb_out], dim=1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class APTInput(BaseModel):
    stat_graph_features: List[float]
    temporal_features: List[float]

app = FastAPI()

@app.post("/predict")
def predict(input_data: APTInput):
    try:
        # Load models only when needed
        lgb_model = joblib.load("lightgbm_model.pkl")

        gru_model = BiGRUModel()
        cnn_model = CNNModel()
        fusion_model = FusionModel()

        checkpoint = torch.load("best_model.pth", map_location="cpu")
        gru_model.load_state_dict(checkpoint['gru'])
        cnn_model.load_state_dict(checkpoint['cnn'])
        fusion_model.load_state_dict(checkpoint['fusion'])

        gru_model.eval()
        cnn_model.eval()
        fusion_model.eval()

        # Prepare inputs
        stat_features = np.array(input_data.stat_graph_features).reshape(1, -1)
        lgb_out_np = lgb_model.predict_proba(stat_features)
        lgb_out_tensor = torch.tensor(lgb_out_np, dtype=torch.float32)

        temporal_input = np.array(input_data.temporal_features).reshape(1, 14, 1)
        temporal_tensor = torch.tensor(temporal_input, dtype=torch.float32)

        gru_out = gru_model(temporal_tensor)
        cnn_input = temporal_tensor.view(1, 1, -1)
        cnn_out = cnn_model(cnn_input)

        output = fusion_model(gru_out, cnn_out, lgb_out_tensor)
        probs = torch.softmax(output, dim=1).detach().numpy()
        predicted_class = int(np.argmax(probs))

        return {
            "prediction": predicted_class,
            "probabilities": probs[0].tolist()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
