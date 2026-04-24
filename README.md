# 🛡️ DeepShield — Explainable Deepfake Audio Detection

## 📌 Project Overview

DeepShield is an AI-powered audio forensics platform designed to detect deepfake (synthetic) audio using machine learning techniques.
It is built for digital forensic investigations with explainability and legal awareness.

The system analyses audio signals and determines whether the voice is **real (bonafide)** or **deepfake**, along with confidence scores and feature explanations.

---

## 🚀 Key Features

* 🔍 **Deepfake Audio Detection**

  * Classifies audio as Real or Deepfake using XGBoost model

* 🧠 **Explainable AI (XAI)**

  * Uses SHAP (SHapley Additive Explanations) to show feature importance

* 🎙️ **Live Audio Recording**

  * Record and analyse audio in real-time

* 📁 **File Upload Support**

  * Supports WAV, MP3, FLAC formats

* ⏱️ **Long Audio Analysis**

  * Splits audio into segments and performs timeline-based detection

* 🎵 **Audio Type Classification**

  * Detects speech, music, noise, scream, animal sounds, etc.

* 📊 **Waveform & Spectrogram Visualization**

* 📄 **Auto PDF Report Generation**

  * Generates forensic report with confidence and legal references

* ⚖️ **Cyber Law Integration**

  * Displays relevant Indian IT Act and IPC sections

---

## 🛠️ Technologies Used

* Python
* Streamlit
* Librosa
* Scikit-learn
* XGBoost
* SHAP
* Matplotlib

---

## 📂 Project Structure

```
DeepShield/
│── app.py
│── extract_features.py
│── deepfake_audio_xgboost_model.pkl
│── scaler.pkl
│── README.md
```

---

## 📊 Model Details

* Dataset: ASVspoof 2019 (Logical Access)

* Features:

  * MFCC (40)
  * Delta MFCC (40)
  * Delta-Delta MFCC (40)
  * Spectral Centroid
  * Zero Crossing Rate
  * Spectral Bandwidth

* Total Features: 123

* Model:

  * XGBoost Classifier
  * Accuracy: ~87%

---

## ⚠️ Limitations

* Works best on human speech audio
* Less accurate for music or environmental sounds
* Requires clean audio for better prediction

---

## 🔮 Future Improvements

* Deep learning models (CNN, LSTM)
* Real-time streaming detection
* Multilingual dataset support
* Mobile app integration

---

## 📜 Disclaimer

This project is for academic and research purposes only.
Always consult legal professionals before taking action based on forensic results.

---
