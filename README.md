# TinyML: CNN Reduction and Quantization for Handwritten Digit Classification

**Mini-project:** Part D — Question 23 (10 marks)  
**Platform:** Google Colab  
**Dataset:** MNIST handwritten digits  
**Frameworks:** TensorFlow / Keras and TensorFlow Lite

## 1. Introduction

Convolutional Neural Networks (CNNs) are widely used for image classification because they learn visual features such as edges, curves, and shapes. However, CNNs can require substantial memory, computation, and storage. These requirements can make deployment difficult on microcontrollers, which typically have limited RAM, flash memory, and processing capability.

Tiny Machine Learning (TinyML) aims to run machine-learning models on small, resource-constrained devices, often with low power consumption and without relying on cloud computing.

This project explores how a CNN that classifies handwritten digits can be made smaller using fewer convolution filters and quantization. The project compares model accuracy, parameter count, and model file size to understand the trade-off between efficiency and classification performance.

## 2. Problem Statement

A CNN trained for handwritten-digit classification may perform well on a computer but may be too large or computationally expensive for a microcontroller. The challenge is to reduce its model complexity and storage requirements while retaining acceptable classification accuracy.

The objective of this mini-project is to investigate whether a digit-classification CNN can be optimized for TinyML by:
- reducing the number of convolution filters;
- applying post-training quantization to the compact model; and
- comparing the original, compact, and quantized models using consistent test data.

The experiment uses the MNIST dataset, which contains grayscale images of handwritten digits from 0 to 9.

## 3. Aim and Objectives

### Aim

To study CNN reduction techniques for TinyML and evaluate their effect on handwritten-digit classification accuracy and model size.

### Objectives

1. Load and preprocess the MNIST handwritten-digit dataset.
2. Train a reference CNN and a compact CNN with fewer convolution filters.
3. convert the compact CNN to TensorFlow Lite with post-training quantization.
4. Evaluate the models on the same test dataset.
5. Compare test accuracy, trainable parameter count, and model file size.
6. Discuss whether the optimized model is a promising candidate for microcontroller deployment and identify what additional hardware measurements would be required.

## 4. Dataset

- **Name:** MNIST handwritten digits
- **Training set:** 60,000 images
- **Test set:** 10,000 images
- **Image format:** 28 × 28 grayscale pixels
- **Classes:** 10 (digits 0 through 9)
- **Loading method:** `tf.keras.datasets.mnist.load_data()`

The notebook downloads/loads the dataset through the TensorFlow/Keras dataset API. The dataset files are not stored in this repository.

**Dataset and documentation links**
- Keras MNIST dataset: https://keras.io/api/datasets/mnist/
- TensorFlow beginner classification tutorial: https://www.tensorflow.org/tutorials/quickstart/beginner
- TensorFlow Lite documentation: https://www.tensorflow.org/lite

Check the dataset's source and license terms if redistributing the data separately.

## 5. Proposed Methodology

The experiment follows these steps:

1. **Load and preprocess data:** Load MNIST, normalize pixel values from 0–255 to 0–1, and reshape each image to include a single channel.
2. **Train the reference CNN:** Train a CNN with a larger number of convolution filters to establish a baseline.
3. **Train the compact CNN:** Train a second CNN with fewer filters and a smaller dense layer.
4. **Quantize the compact model:** Convert it to TensorFlow Lite using post-training quantization and a representative sample of training images.
5. **Evaluate:** Measure test accuracy for both Keras models and run inference with the quantized TensorFlow Lite model.
6. **Compare:** Record accuracy, parameter count, and saved model size, and inspect some misclassified images.
7. **Discuss TinyML suitability:** Explain the observed trade-off and the additional measurements needed on a physical microcontroller.

## 6. CNN Architectures

| Layer | Reference CNN | Compact CNN |
|---|---|---|
| Input | 28 × 28 × 1 | 28 × 28 × 1 |
| Conv2D, 3 × 3, ReLU | 16 filters | 8 filters |
| MaxPooling2D, 2 × 2 | Yes | Yes |
| Conv2D, 3 × 3, ReLU | 32 filters | 16 filters |
| MaxPooling2D, 2 × 2 | Yes | Yes |
| Flatten | Yes | Yes |
| Dense, ReLU | 64 units | 32 units |
| Output | 10 units, softmax | 10 units, softmax |

The reference CNN provides a baseline. The compact CNN uses fewer filters and fewer dense units to reduce model complexity. Both models are trained and tested on the same dataset split for a meaningful comparison.

## 7. Optimization Techniques

### 7.1 Reducing the Number of Filters

Convolution filters learn features from the input images. Reducing the filter counts from 16/32 to 8/16 decreases the number of feature detectors and generally reduces parameters and computation. However, fewer filters may limit the network's ability to learn complex patterns, which can reduce accuracy.

### 7.2 Quantization

Post-training quantization converts a trained model into a more deployment-oriented representation. This notebook requests INT8 operations for supported internal operations while keeping the input and output tensors in float32 for easier evaluation.

Quantization can reduce model storage and may improve inference efficiency on hardware that supports integer operations. The exact benefit depends on the model, conversion settings, runtime, and target hardware. Quantization may also cause a small change in predictions or accuracy.

### 7.3 Accuracy–Efficiency Trade-off

The aim is not simply to achieve the highest accuracy. The goal is to find a practical balance between:
- classification accuracy;
- model file size and flash storage;
- RAM requirements;
- inference latency; and
- energy consumption.

This notebook measures accuracy, parameter count, and model file size. It does **not** measure actual microcontroller RAM use, latency, or power consumption.

## 8. How to Run in Google Colab

1. Download `TinyML_CNN_MNIST_Quantization.ipynb` from this repository.
2. Open Google Colab: https://colab.research.google.com/
3. Select **File → Upload notebook** and upload the notebook.
4. Connect to a runtime. A GPU is optional for this small experiment.
5. Select **Runtime → Run all**.
6. Wait for the dataset to load, the two models to train, and the quantized model to be evaluated.
7. Review the printed metrics and plots.
8. Download the executed notebook using **File → Download → Download .ipynb** if you want to preserve outputs for submission.

Execution time depends on runtime startup, internet speed, and available compute resources. Use the actual values from your run in the report.

## 9. Results

Fill in the table using the results produced by your notebook. Do not report unmeasured or assumed values.

| Model | Test accuracy | Trainable parameters | File size |
|---|---:|---:|---:|
| Reference CNN (Keras) | Record result | Record result | Record result |
| Compact CNN (Keras) | Record result | Record result | Record result |
| Compact CNN (TFLite float32) | Record result | Record result | Record result |
| Compact CNN (TFLite quantized) | Record result | Record result | Record result |

The notebook generates comparison plots and saves a metrics file at `artifacts/model_comparison.csv` in the Colab runtime. If you want to include this file in the GitHub repository, download it from Colab and upload it separately.

## 10. Discussion Guide

After running the notebook, use your measured results to discuss:

- Did the compact CNN use fewer trainable parameters than the reference CNN?
- How did the compact CNN's accuracy compare with the reference model?
- How much did the TensorFlow Lite quantized model's file size change relative to the float model?
- Did quantization change test accuracy? If so, by how much?
- Based on the results, what advantages and limitations would the optimized model have on a microcontroller?

Reducing filters generally decreases model complexity, while quantization reduces numerical precision and can reduce storage. Both methods may affect accuracy. The best configuration depends on the application's accuracy requirements and the target device's memory and compute limits.

## 11. Conclusion

This mini-project investigates two practical techniques for preparing a CNN for TinyML: reducing convolution filters and applying post-training quantization. The reference, compact, and quantized models are compared using MNIST test accuracy, parameter count, and model file size.

Complete the conclusion after running the experiment. State whether the compact and quantized models maintained acceptable accuracy while reducing model complexity or storage, and support the conclusion with the measured results.

**Scope limitation:** The experiment is performed in Google Colab. Successful conversion to TensorFlow Lite does not by itself prove deployment on a microcontroller. A real deployment would require testing operator compatibility and measuring flash usage, peak RAM, inference latency, and power or energy consumption on the selected device.

## 12. Repository Structure

```text
tinyml-cnn-mnist/
├── README.md
├── TinyML_CNN_MNIST_Quantization.ipynb
└── requirements.txt
```

The dataset is loaded by the notebook and is not included in the repository. Model files and comparison outputs are generated during execution and can be uploaded if required by the instructor.

## 13. References

1. Keras, “MNIST digits classification dataset.” https://keras.io/api/datasets/mnist/
2. TensorFlow, “TensorFlow 2 quickstart for beginners.” https://www.tensorflow.org/tutorials/quickstart/beginner
3. TensorFlow, “TensorFlow Lite.” https://www.tensorflow.org/lite
4. GitHub Docs, “Adding a file to a repository.” https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository

## 14. Author

- **Name:** Add your name
- **Course:** M.E. Embedded Systems / add the appropriate course
- **Institution:** Add your institution
- **Subject / Assignment:** CNN Mini-project — Part D, Question 23
