# 🕵️‍♀️ Steganalysis Detector

Uncover the invisible\! This project presents a powerful CNN-based steganalysis tool designed to detect hidden data embedded within images, complete with an intuitive Streamlit web application for real-time analysis.

## 🌟 Features

  * **CNN Model:** A deep Convolutional Neural Network specifically designed to capture the subtle, high-frequency noise patterns introduced by steganography, enabling robust detection.
  * **Dataset Handling:** Optimized for Kaggle's "StegoImagesDataset" (`train`/`val`/`test` splits with `clean`/`stego` subdirectories).
  * **Class Imbalance:** Uses class weighting during training.
  * **Streamlit Web App:** Interactive UI for image upload and prediction.
  * **Evaluation:** Reports accuracy, precision, and recall.

## 💡 Project Overview

In an era where digital communication is pervasive, steganography poses a subtle threat by allowing malicious data to be concealed within seemingly innocuous images. This project addresses this challenge by training a CNN to rigorously distinguish between original ('clean') and steganographic ('stego') images, providing a crucial tool for digital forensics and security via an intuitive Streamlit interface.

## 📸 Demo / Screenshots

[Screenshot of Streamlit app in action]
[Screenshot of prediction output]

## 📊 Dataset

Uses the [**StegoImagesDataset**](https://www.kaggle.com/datasets/marcozuppelli/stegoimagesdataset/data) from Kaggle.
**Structure:** `/kaggle/input/stegoimagesdataset/[train|val|test]/[train|val|test]/[clean|stego]/`

## 🛠️ Technologies

  * Python 3.x
  * TensorFlow / Keras
  * NumPy, Pillow, Scikit-learn
  * Streamlit


## 📈 Results and Performance

Model performance depends on subtle data patterns. Class weights help balance evaluation. Key metrics: Accuracy, Precision, Recall. Further tuning and advanced architectures can improve results.

## 🚀 Future Enhancements

  * Advanced CNN architectures.
  * Specialized data augmentation.
  * Detection of diverse steganography methods.
  * Explainable AI (XAI) integration.
  * Alternative deployment options.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.
