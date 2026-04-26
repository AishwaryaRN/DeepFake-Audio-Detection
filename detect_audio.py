import librosa
import numpy as np
import joblib

# ===== LOAD TRAINED MODEL =====
model = joblib.load("deepfake_audio_xgboost_model.pkl")
print("Model loaded successfully!")

# ===== AUDIO FILE PATH =====
audio_path = "test_audio1.wav"   # change if needed

# ===== LOAD AUDIO =====
audio, sr = librosa.load(audio_path, sr=16000)  # force consistent sample rate

# ===== NORMALIZE AUDIO (IMPORTANT) =====
audio = audio / np.max(np.abs(audio))

# ===== EXTRACT MFCC =====
mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
mfcc_mean = np.mean(mfcc.T, axis=0)

# reshape for model
mfcc_mean = mfcc_mean.reshape(1, -1)

# ===== PREDICT =====
prediction = model.predict(mfcc_mean)
prob = model.predict_proba(mfcc_mean)

# ===== RESULT =====
if prediction[0] == 0:
    print("\nPrediction: Bonafide (Real Audio)")
else:
    print("\nPrediction: Deepfake Audio")

print("Confidence:", prob)