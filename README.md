# Emotion Detection System: Smile Detection

## Overview

This project aims to develop an emotion recognition system focused on smile detection through facial expression analysis. It combines face detection using the `dlib` library with a Convolutional Neural Network (CNN) trained on facial expression data. The system accurately detects faces in images, preprocesses them, and classifies expressions as "Happy", "Sad", or "Neutral".

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model Architecture](#model-architecture)
- [Evaluation](#evaluation)
- [Face Detection with dlib](#face-detection-with-dlib)
- [Results](#results)
- [Conclusion](#conclusion)
- [References](#references)

## Dataset

The model is trained on the [FER-2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) dataset, which contains 48x48 grayscale facial images categorized into seven emotions. For improved performance and balance, only the three most prevalent emotions were selected:

- Happy (3)  
- Sad (4)  
- Neutral (6)

The dataset consists of:
- 28,709 training samples
- 3,589 public test samples

## Preprocessing

- Only three classes (Happy, Sad, Neutral) were used to handle class imbalance.
- Images were converted from pixel strings to grayscale image arrays.
- Labels were one-hot encoded.
- Data was split into 90% training and 10% testing using `train_test_split`.
- All image pixel values were normalized.

## Model Architecture

The deep CNN was adapted from a public implementation by Aayush Mishra, with significant enhancements:
- ReLU activations were replaced by ELU for better convergence.
- The number of filters was increased (64, 128, 256).
- Larger kernel sizes (5x5 followed by 3x3) were used to capture both broad and fine features.
- Dropout rates increased to 0.4–0.5 to prevent overfitting.
- L2 regularization was removed in favor of batch normalization and dropout.

### Callbacks Used

- **EarlyStopping**: Stops training when validation accuracy plateaus.
- **ReduceLROnPlateau**: Reduces learning rate when improvements slow.

## Evaluation

### Accuracy and Loss

- Validation accuracy stabilized at ~80% after 40 epochs.
- Validation loss closely followed training loss, indicating no overfitting.

### Confusion Matrix

- High accuracy in identifying "Happy".
- Some confusion between "Sad" and "Neutral".

### Final Test Results

- **Test Loss**: 0.457  
- **Test Accuracy**: 82.04%

## Face Detection with dlib

The `dlib` library is employed to:
1. Detect and isolate faces in input images.
2. Resize the detected face to 48x48 pixels.
3. Convert it to grayscale.
4. Reshape it to a tensor format compatible with the Keras model: `(1, 48, 48, 1)`.

## Results

The model performed well in recognizing emotions across the selected categories. Most predictions aligned with the ground truth, showcasing the robustness of the preprocessing pipeline and model design.

## Conclusion

This project demonstrates the effectiveness of combining classic face detection with modern deep learning architectures for real-time emotion recognition. By focusing on data balancing, preprocessing consistency, and model optimization, the system achieved high accuracy in classifying facial expressions. Iterative experimentation with CNN parameters and callbacks significantly enhanced model performance.

## References

1. Scikit-learn: [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)  
2. FER-2013 Dataset on Kaggle: [Facial Expression Challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)  
3. Mishra, Aayush. [Emotion Detector](https://www.kaggle.com/code/aayushmishra1512/emotion-detector)
