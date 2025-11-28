import torch
import pandas as pd
from model import HeartRateAnomalyDetector

# Load data
df = pd.read_csv("data/sample_heart_rate.csv")
values = torch.tensor(df['heart_rate'].values, dtype=torch.float32).unsqueeze(1)

# Load model
model = HeartRateAnomalyDetector()
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
model.eval()

# Predict anomalies
with torch.no_grad():
    preds = model(values)
    diffs = torch.abs(preds - values)
    anomalies = diffs > 15  # threshold

df['prediction'] = preds.squeeze().numpy()
df['anomaly'] = anomalies.squeeze().numpy()

print(df[df['anomaly'] == 1])
