# Rommel_Tuquerres_lab15 · Medical Image Recognition (Android + ML Kit)

Android app that classifies pediatric chest X-rays as **normal** or **pneumonia** on the device,
using a custom TensorFlow Lite model with **ML Kit Custom Image Labeling**.

**Student:** Rommel Tuquerres · Artificial Intelligence · Universidad Yachay Tech · LAB 15

## Model
| Item | Detail |
|---|---|
| Dataset | PneumoniaMNIST (MedMNIST v3), 128×128 px · train 4,708 · val 524 · test 624 |
| Architecture | MobileNetV2 (ImageNet) · transfer learning + fine-tuning (last 40 layers) |
| Test accuracy | **0.832** (Keras and TFLite) · validation 0.945 |
| Recall | pneumonia 0.982 · normal 0.581 |
| Export | `model.tflite` with metadata (normalization + labels), 8.7 MB |

Training notebook: `Rommel_Tuquerres_lab15.ipynb` (TensorFlow 2.21, local VS Code, CPU).

## App
- **Home** with access to two sections.
- **Model:** dataset, architecture, accuracy, training curves and confusion matrix.
- **Predictions:** 20 random test X-rays with true label, phone prediction and notebook prediction;
  "Classify the whole test set" shows the phone accuracy (15/20) and agreement with the notebook.
  Images can also be picked from the phone gallery.

## Run
1. Open the project in Android Studio.
2. Connect an Android device (USB debugging) and press **Run**.

Assets: `app/src/main/assets/` → `model.tflite`, `test_set/` (20 images + `labels.csv`), training plots.

Dependency: `com.google.mlkit:image-labeling-custom:17.0.3`

Based on the Google codelab *Integrate a Custom Model into your App*.
