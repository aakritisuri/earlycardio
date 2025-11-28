# Model Card — EarlyCardio LSTM Heart Rate Anomaly Detector

## 📌 Model Overview  
This model predicts the next heart-rate value from a sliding window of the previous 5 minutes of data using a one-layer LSTM built in PyTorch. Anomalies are detected by comparing actual vs predicted values and flagging predictions whose residuals exceed a statistical threshold.

---

## 🎯 Intended Use  
- Educational and exploratory use  
- Demonstration of time-series modeling with PyTorch  
- Prototype for early cardiovascular anomaly detection  
- Suitable for research and experimentation  

---

## 🚫 Not Intended For  
- Clinical diagnosis  
- Emergency response  
- Real-world medical decision-making  
- Deployment without regulatory clearance and medical oversight  

---

## ⚙ Architecture  
- **Model**: LSTM (1 layer, hidden size 32)  
- **Input**: 5-step HR window (resampled to 1-minute intervals)  
- **Output**: regression estimate for next HR value  
- **Anomaly method**: residual-based thresholding (mean + 2.5σ)

---

## 📊 Dataset  
A synthetic heart-rate dataset with two tachycardia-like spikes.  
Filename: `data/sample_heart_rate.csv`.

Synthetic data ensures reproducibility without requiring private wearable data.

---

## ⚠ Limitations  
- Not validated on clinical datasets  
- Doesn’t account for multi-signal inputs (HRV, SpO₂, respiration)  
- Performance limited by small dataset  
- Threshold-based anomaly detection = simple, not clinical-grade  
- Not personalized to age, gender, fitness level, or condition  

---

## 🤝 Ethical Considerations  
- False positives may cause unnecessary concern  
- False negatives may miss meaningful events  
- Not designed or approved for any medical use  
- Real deployments would require large-scale validation, fairness checks, and regulatory pathways  

---

## 💡 Future Improvements  
- Incorporate HRV and multiple biosignals  
- Train on larger wearable datasets  
- Move from residual thresholding to probabilistic anomaly scoring  
- Deploy on edge devices for low-bandwidth populations  
- Integrate into preventative-care platforms  

