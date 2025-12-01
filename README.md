# 🚀 Bitcoin Price Prediction Web Application using LSTM & GRU (PyTorch + Flask)

This system supports multi-day forecasting, professional visualization, real-time API usage, and automatic fallback to demo mode when models are unavailable.

---

## 🖼️ Application Preview

> Replace the image paths with your real screenshots.

<p align="center">
  <img src="Demo/1.png" width="45%"/>
  <img src="Demo/2.png" width="45%"/>
</p>

<p align="center">
  <img src="Demo/3.png" width="45%"/>
  <img src="Demo/4.png" width="45%"/>
</p>

<p align="center">
  <img src="Demo/5.png" width="45%"/>
</p>
---

## 📌 Key Features

- ✅ Bitcoin price forecasting using **LSTM & GRU**
- ✅ Built with **PyTorch**
- ✅ Flask-based web interface & REST API
- ✅ Multi-horizon forecasting: **7, 15, 30, 60, 90 days**
- ✅ Automatic **scaler loading**
- ✅ Professional prediction summary statistics
- ✅ GPU support if available
- ✅ Demo (mock) mode if models are missing
- ✅ Clean MVC project structure

---

## 🧠 Deep Learning Models

### 🔹 LSTM (Long Short-Term Memory)
- Input: 1D Time Series
- Hidden Size: 50
- Layers: 2
- Dropout: 0.2
- Output: 1 value per step

### 🔹 GRU (Gated Recurrent Unit)
- Input: 1D Time Series
- Hidden Size: 50
- Layers: 2
- Dropout: 0.2
- Output: 1 value per step

Both models are trained offline and loaded as `.pth` PyTorch weights during runtime.

---

## 🏗️ Project Structure

