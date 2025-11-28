# earlycardio  
### **LSTM-Based Anomaly Detection for Preventive Cardiac Monitoring**  
*A lightweight ML pipeline that detects abnormal heart-rate events from noisy, sparse wearable data using PyTorch.*

---

## Project Overview
Cardiovascular deaths often occur without warning because early signals go unnoticed. This project builds a **PyTorch LSTM anomaly detector** that identifies abnormal heart-rate spikes (e.g., tachycardia-like events) from timestamped sensor data.

The system ingests raw heart-rate readings → learns normal physiological patterns → flags deviations indicative of potential cardiovascular risk.

This demonstrates:
- time-series modeling (LSTM)
- signal preprocessing + normalization  
- sliding-window sequence generation  
- reconstruction-based anomaly scoring  
- visualization + interpretability  
- modular ML engineering in a clean repo structure

All code runs end-to-end in `earlycardio.ipynb` with saved model weights (`model.pth`).

---

## Why This Matters
Cardiovascular disease kills young people silently — often in settings with:
- intermittent data  
- low health literacy  
- lack of preventive care  
- no clinical-grade monitoring  

This model is intentionally designed for **low-data**, **low-connectivity**, and **edge-friendly** environments like India.

It forms the foundation of a preventive health system built around:  
**early detection using minimal data for populations that cannot access continuous monitoring.**

---

## System Architecture

### **1. Data Pipeline**
Input CSV format:
```python
timestamp, heart_rate
2025-01-01 00:00:00, 72
2025-01-01 00:01:00, 73
...
```
Processing steps:
- timestamp parsing  
- normalization  
- sliding-window generation (`seq_len = 10`)  
- train/test split  

---

### **2. Model**
A compact LSTM model:

```python
class LSTMHeartRateModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)
```

Why LSTM (vs transformer models)?
- strong performance on small datasets
- low parameter count
- deployment-friendly
- ideal for sequential physiological datasets

### **3. Training**
- Loss: MSE
- Optimizer: Adam
- Epochs: 25
- Data: sliding-window sequences of heart-rate values
- Goal: reconstruct normal heart-rate behavior

### **4. Anomaly Detection Logic**

Anomalies are flagged using reconstruction error.

**Threshold:**
```python
mean(error) + 2 × std(error)
```
Any sample crossing this threshold → potential cardiac anomaly.

### **Example Output**

**Anomaly Detection Plot**
(Located at: results/anomaly_plot.png)

Red points represent detected tachycardia-like spikes.

### Example: Load Data + Run Anomaly Detection

Below is a minimal reproducible snippet showing how to load the sample dataset,
run the trained LSTM model, compute anomaly scores, and visualize detected spikes.

```python
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# -------------------------
# 1. Load and preprocess data
# -------------------------
df = pd.read_csv("sample_heart_rate.csv", parse_dates=["timestamp"])
values = df["heart_rate"].values.reshape(-1, 1)

scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(values)

# Create sliding windows
seq_len = 10
X = []
for i in range(len(scaled_values) - seq_len):
    X.append(scaled_values[i:i+seq_len])
X = np.array(X)

X_tensor = torch.tensor(X, dtype=torch.float32)

# -------------------------
# 2. Load trained model
# -------------------------
from lstm_model import LSTMHeartRateModel  # if model class is in notebook, redefine it here

model = LSTMHeartRateModel(input_dim=1, hidden_dim=32)
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

# -------------------------
# 3. Forward pass + anomaly scoring
# -------------------------
with torch.no_grad():
    preds = model(X_tensor).numpy()

# Compute reconstruction error
errors = np.abs(preds.flatten() - scaled_values[seq_len:].flatten())

# Threshold for anomaly detection
threshold = np.mean(errors) + 2 * np.std(errors)
anomalies = errors > threshold

# -------------------------
# 4. Visualization
# -------------------------
plt.figure(figsize=(12, 5))
plt.plot(df["timestamp"][seq_len:], values[seq_len:], label="Heart Rate")
plt.scatter(
    df["timestamp"][seq_len:][anomalies],
    values[seq_len:][anomalies],
    color="red",
    label="Anomaly",
)
plt.legend()
plt.title("Heart-Rate Anomaly Detection")
plt.xlabel("Time")
plt.ylabel("Heart Rate (bpm)")
plt.show()

```

### **Repository Structure**
```python
earlycardio/
│
├── earlycardio.ipynb            # End-to-end training + inference notebook
├── model.pth                     # Trained LSTM weights
├── sample_heart_rate.csv         # Demo dataset
├── ModelCard.md                  # Model overview + evaluation
├── requirements.txt              # Reproducible dependencies
└── results/
    └── anomaly_plot.png          # Output visualization
```
### **Reproducibility**
1. Install dependencies:
pip install -r requirements.txt
2. Open earlycardio.ipynb
3. Run all cells
4. Load pretrained weights using:
model.load_state_dict(torch.load("model.pth")); model.eval()

### **Engineering Decisions**

**Designed for Low-Resource Environments**
- CPU-friendly
- <100k parameters
- Handles sparse, noisy signals
- Works offline
- Easy mobile/edge deployment

**Clean ML Engineering**
- Modular data loader
- Reusable model architecture
- Clear anomaly scoring
- Visual outputs for interpretability

**Production Aware**
- Model weights saved
- Requirements pinned
- ModelCard included
- Fully reproducible pipeline

### **Future Work**

Planned extensions:
- HRV-based modeling
- sleep-aware anomaly detection
- arrhythmia prediction
- personalized baseline adaptation
- multi-sensor fusion (SpO2, motion, stress)
- quantized deployment (TFLite, CoreML, ONNX)
- API-first ecosystem for health-app integration
