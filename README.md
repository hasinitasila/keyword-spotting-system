# Keyword Spotting (KWS) Using MFCC and Convolutional Neural Networks

## Problem Statement
Design and implement a **Keyword Spotting (KWS)** system that can detect predefined spoken keywords from audio signals using **Mel-Frequency Cepstral Coefficients (MFCC)** and **Convolutional Neural Networks (CNNs)**.  
The system should support both **offline training** and **real-time keyword detection** using microphone input.

---

## 1. Implementation

### 1.1 Programming Language and Tools Used
The complete implementation of the Keyword Spotting system was carried out using **Python**.
**Tools and Libraries:**
- **TensorFlow / Keras** – CNN model building and training  
- **Librosa** – Audio preprocessing and MFCC feature extraction  
- **NumPy** – Numerical computation  
- **Scikit-learn** – Dataset splitting and evaluation  
- **SoundDevice** – Real-time microphone audio capture  
- **Matplotlib** – Visualization of training performance  

The model was trained and evaluated using the **Google Speech Commands Dataset**, which consists of short spoken words sampled at **16 kHz**.

---

## 1.2 Methodology Overview
The objective of the system is to identify predefined keywords from speech signals using MFCC features and a CNN-based classifier.  
The system workflow includes audio preprocessing, feature extraction, model training, evaluation, and real-time deployment.

---

## 1.3 Algorithm

### Step 1: Audio Input Acquisition
- Training audio samples are taken from the Speech Commands dataset.
- During real-time testing, audio is captured from the microphone in **1-second segments**.

### Step 2: Audio Preprocessing
- All audio signals are resampled to **16 kHz**.
- Signals are padded or truncated to **1 second (16000 samples)** to ensure uniform input size.

### Step 3: MFCC Feature Extraction
- MFCCs are extracted using **Librosa**.
- MFCCs capture perceptually relevant frequency information and are widely used in speech recognition tasks.

### Step 4: Dataset Preparation
- MFCC features are stored as **2D matrices**.
- Each sample is labeled with its corresponding keyword.
- Data is split into **training, validation, and test sets**.

### Step 5: CNN Model Design and Training
- A CNN is used to learn spatial patterns from MFCC feature maps.
- Architecture includes:
  - Convolution layers
  - Batch normalization
  - Max pooling
  - Fully connected layers
  - Softmax output layer for multi-class classification

### Step 6: Model Evaluation
- Performance is evaluated using:
  - Classification accuracy
  - Confusion matrix
  - Precision and recall metrics

### Step 7: Real-Time Keyword Spotting
- The trained model is deployed for real-time inference.
- Audio is processed on-the-fly and passed through the CNN.
- A **confidence threshold** and **prediction smoothing window** reduce false detections.

---

## 1.4 Code Description

### `train.ipynb`
- Downloads and preprocesses the Speech Commands dataset
- Extracts MFCC features
- Trains a CNN-based classifier
- Evaluates performance using accuracy and confusion matrix
- Saves the trained model as `kws_mfcc_cnn.h5`

### `realtime_test.py`
- Captures live audio from the microphone
- Extracts MFCC features in real time
- Loads the trained CNN model
- Predicts and displays detected keywords with confidence scores

---

## 2. Results

The CNN-based keyword spotting system demonstrated strong performance on the Speech Commands dataset.

- Training and validation accuracy showed stable convergence
- Confusion matrix exhibited strong diagonal dominance, indicating correct classification of most keywords
- Real-time testing achieved **low-latency detection**
- MFCC features combined with confidence thresholding significantly reduced false detections caused by background noise

Overall, the results confirm that the **MFCC + CNN** approach is effective and suitable for real-time keyword spotting applications.

---

## 3. References

[1] TensorFlow, “Speech Commands Dataset.”  
[2] D. P. Ellis, “PLP and RASTA and MFCC,” Columbia University, 2005.  
[3] Librosa Development Team, “Librosa: Audio and Music Signal Analysis in Python.”  
[4] Google, “TinyML Keyword Spotting Example.”  
[5] I. Goodfellow, Y. Bengio, and A. Courville, *Deep Learning*, MIT Press, 2016.

---
