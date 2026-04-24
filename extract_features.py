import os
import librosa
import numpy as np
import random
import joblib
import shap
import matplotlib.pyplot as plt
import noisereduce as nr
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# ===== PATHS =====
protocol_path = r"C:\Users\AISWARYA\Desktop\DeepFake Audio\dataset\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt"
audio_dir     = r"C:\Users\AISWARYA\Desktop\DeepFake Audio\dataset\LA\ASVspoof2019_LA_train\flac"

# ===== FIX 2: YOUR OWN VOICE RECORDINGS =====
# Place your 20-30 recorded .wav files in this folder
# Record them with record_my_voice.py (provided separately)
own_voice_dir = r"C:\Users\AISWARYA\Desktop\DeepFake Audio\my_voice_samples"

# ===== READ LABELS =====
labels = {}

with open(protocol_path, "r") as f:
    for line in f:
        parts   = line.strip().split()
        file_id = parts[1].strip()
        label   = parts[-1]
        labels[file_id] = 0 if label == "bonafide" else 1

print("Total labels loaded:", len(labels))

# ===== BALANCE DATASET =====
file_list      = list(labels.keys())
random.shuffle(file_list)

bonafide_files = [f for f in file_list if labels[f] == 0]
spoof_files    = [f for f in file_list if labels[f] == 1]

sample_size    = 2000
selected_files = bonafide_files[:sample_size] + spoof_files[:sample_size]
random.shuffle(selected_files)

print("Selected files:", len(selected_files))
print("Bonafide files:", len(bonafide_files))
print("Spoof files   :", len(spoof_files))


# =====================================================
# SHARED FEATURE EXTRACTION
# Returns 123-dim vector or None if audio is silent
# =====================================================
def extract_features_from_audio(audio, sr):
    """
    Feature vector layout (123 total):
      [0:40]   MFCC mean
      [40:80]  Delta-MFCC mean      <- FIX 4: captures voice dynamics
      [80:120] Delta-delta-MFCC mean <- FIX 4: captures voice acceleration
      [120]    Spectral centroid mean
      [121]    ZCR mean
      [122]    Spectral bandwidth mean
    """

    # FIX 1: Noise reduction — same as app.py, so scaler stays consistent
    audio = nr.reduce_noise(y=audio, sr=sr)

    # FIX 3: Energy gating — reject silent / near-silent audio
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 0.01:
        return None

    # Normalize amplitude
    audio = audio / (np.max(np.abs(audio)) + 1e-6)

    # Fix length to exactly 3 seconds
    max_len = 3 * sr
    if len(audio) > max_len:
        audio = audio[:max_len]
    else:
        audio = np.pad(audio, (0, max_len - len(audio)))

    # ===== MFCC =====
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

    # FIX 4: Delta and delta-delta MFCC
    # Real voices have natural pitch/tempo variation → higher delta values
    # Synthetic voices are smoother → delta values are lower/flatter
    mfcc_delta  = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    mfcc_mean       = np.mean(mfcc.T,        axis=0)  # (40,)
    mfcc_delta_mean = np.mean(mfcc_delta.T,  axis=0)  # (40,)
    mfcc_d2_mean    = np.mean(mfcc_delta2.T, axis=0)  # (40,)

    # ===== SPECTRAL FEATURES =====
    spectral_centroid  = np.mean(librosa.feature.spectral_centroid(y=audio,  sr=sr))
    zcr                = np.mean(librosa.feature.zero_crossing_rate(audio))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))

    # ===== COMBINE: 40 + 40 + 40 + 1 + 1 + 1 = 123 =====
    combined = np.hstack([
        mfcc_mean,
        mfcc_delta_mean,
        mfcc_d2_mean,
        spectral_centroid,
        zcr,
        spectral_bandwidth
    ])

    return combined


# =====================================================
# PROCESS ASVSPOOF DATASET
# =====================================================
features = []
y_labels = []
skipped  = 0

for file_id in selected_files:

    file_path = os.path.join(audio_dir, file_id + ".flac")

    if not os.path.exists(file_path):
        print("Missing:", file_path)
        skipped += 1
        continue

    audio, sr = librosa.load(file_path, sr=16000)
    feat = extract_features_from_audio(audio, sr)

    if feat is None:
        print("Skipped (silent):", file_id)
        skipped += 1
        continue

    features.append(feat)
    y_labels.append(labels[file_id])

print(f"\nDataset done. Loaded: {len(features)}, Skipped: {skipped}")


# =====================================================
# FIX 2: ADD YOUR OWN VOICE SAMPLES
# All files in own_voice_dir are labelled bonafide (0)
# =====================================================
own_voice_count = 0

if os.path.exists(own_voice_dir):
    own_files = [f for f in os.listdir(own_voice_dir) if f.endswith(".wav")]
    print(f"\nFound {len(own_files)} own-voice samples → {own_voice_dir}")

    for fname in own_files:
        fpath     = os.path.join(own_voice_dir, fname)
        audio, sr = librosa.load(fpath, sr=16000)
        feat      = extract_features_from_audio(audio, sr)

        if feat is None:
            print("Skipped (silent):", fname)
            continue

        features.append(feat)
        y_labels.append(0)   # bonafide
        own_voice_count += 1

    print(f"Added {own_voice_count} own-voice samples as bonafide.")

else:
    print(f"\nOwn-voice folder not found at: {own_voice_dir}")
    print("Create the folder and add 20-30 WAV recordings of yourself.")
    print("Use record_my_voice.py to record them.")


# =====================================================
# NUMPY ARRAYS + SCALING
# =====================================================
X = np.array(features)
y = np.array(y_labels)

print("\nFeature vector shape:", X.shape)   # (N, 123)
print("Bonafide samples    :", sum(y == 0))
print("Spoof samples       :", sum(y == 1))

scaler = StandardScaler()
X      = scaler.fit_transform(X)
joblib.dump(scaler, "scaler.pkl")
print("Scaler saved → scaler.pkl")


# =====================================================
# TRAIN / TEST SPLIT
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================================
# MODELS
# =====================================================
rf_model  = RandomForestClassifier(n_estimators=100)
gb_model  = GradientBoostingClassifier()
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss"
)

print("\nTraining models...")
rf_model.fit(X_train,  y_train)
gb_model.fit(X_train,  y_train)
xgb_model.fit(X_train, y_train)

joblib.dump(rf_model,  "rf_model.pkl")
joblib.dump(gb_model,  "gb_model.pkl")
joblib.dump(xgb_model, "deepfake_audio_xgboost_model.pkl")
print("All models saved.")


# =====================================================
# EVALUATE
# =====================================================
rf_pred  = rf_model.predict(X_test)
gb_pred  = gb_model.predict(X_test)
xgb_pred = xgb_model.predict(X_test)

print("\nMODEL COMPARISON")
print("----------------------------")
print("Random Forest Accuracy    :", accuracy_score(y_test, rf_pred))
print("Gradient Boosting Accuracy:", accuracy_score(y_test, gb_pred))
print("XGBoost Accuracy          :", accuracy_score(y_test, xgb_pred))

rf_prob  = rf_model.predict_proba(X_test)[:, 1]
gb_prob  = gb_model.predict_proba(X_test)[:, 1]
xgb_prob = xgb_model.predict_proba(X_test)[:, 1]

rf_fpr,  rf_tpr,  _ = roc_curve(y_test, rf_prob)
gb_fpr,  gb_tpr,  _ = roc_curve(y_test, gb_prob)
xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_prob)

rf_auc  = auc(rf_fpr,  rf_tpr)
gb_auc  = auc(gb_fpr,  gb_tpr)
xgb_auc = auc(xgb_fpr, xgb_tpr)

print("\nAUC Scores")
print("Random Forest AUC    :", rf_auc)
print("Gradient Boosting AUC:", gb_auc)
print("XGBoost AUC          :", xgb_auc)

plt.figure(figsize=(8, 6))
plt.plot(rf_fpr,  rf_tpr,  label=f"Random Forest (AUC={rf_auc:.3f})")
plt.plot(gb_fpr,  gb_tpr,  label=f"Gradient Boosting (AUC={gb_auc:.3f})")
plt.plot(xgb_fpr, xgb_tpr, label=f"XGBoost (AUC={xgb_auc:.3f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Deepfake Audio Detection")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150)
plt.show()


# =====================================================
# SHAP EXPLAINABILITY (with named features)
# =====================================================
print("\nGenerating SHAP summary plot...")

feature_names = (
    [f"MFCC_{i+1}"    for i in range(40)] +
    [f"dMFCC_{i+1}"   for i in range(40)] +
    [f"d2MFCC_{i+1}"  for i in range(40)] +
    ["SpectralCentroid", "ZCR", "SpectralBandwidth"]
)

explainer   = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, feature_names=feature_names)

print("\nTraining complete!")
