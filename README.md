# EarlyCardio: LSTM-Based Heart Rate Anomaly Detector  
*A lightweight ML pipeline to predict heart-rate patterns and flag arrhythmia-like anomalies using PyTorch.*

## 🚀 Overview  
EarlyCardio is a simple, production-style ML system that models heart-rate time-series data using an LSTM network and detects anomalies based on prediction residuals. It demonstrates:

- Time-series preprocessing  
- Sliding window sequence modeling  
- LSTM regression in PyTorch  
- Residual-based anomaly detection  
- Visualization of arrhythmia-like spikes  
- Full ML workflow in under 100 lines of code  

This project is part of my ongoing work in **preventative cardiology**, exploring how sparse physiological signals can reveal early warning patterns in cardiovascular health.

---

## ⭐ Features
- Predicts next-step heart rate from 5-minute sliding windows  
- Detects anomalies using dynamic residual thresholds  
- Works with low-frequency (1-minute) heart-rate data  
- Fully runnable on CPU in under 10 seconds  
- Lightweight enough for future edge deployment  
- Includes a synthetic HR dataset  

---

## 🧠 Tech Stack
- Python 3  
- PyTorch  
- NumPy  
- Pandas  
- Matplotlib  
- Scikit-learn  

---

## 📊 Example Output

![Anomaly Plot](results/anomaly_detection_plot.png)

*Red points represent detected anomaly spikes (e.g., tachycardia events).*

---

## 🛠 How to Run

### Install dependencies  

### Train the model  

### Run anomaly detection  

### View results  
Check:  

---

## 🧪 Dataset  
A synthetic dataset (`sample_heart_rate.csv`) is included for demonstration.  
Replace it with real heart-rate or wearable data for experimentation.

---

## 🧬 Motivation  
Several relatives in my own family suffered fatal cardiac events without early detection. This reinforced my belief that early, algorithmic screening tools should be accessible even in low-resource settings. This project is my first step toward building **affordable, preventative cardiovascular ML infrastructure**.

---

## 📄 Model Card  
See **ModelCard.md** for intended use, ethics, and limitations.



