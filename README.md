# Traffic Sign Recognition using Convolutional Neural Networks and Tensorflow
This repository contains a Jupyter notebook focusing on traffic sign recognition using Convolutional Neural Networks (CNNs) implemented using TensorFlow and Keras.

## Overview
The provided Jupyter notebook demonstrates the process of building and training various CNN architectures for traffic sign recognition. The notebook covers the following aspects:

* Data loading and preprocessing: Loading the traffic sign dataset and preprocessing the images for training.
* Architecture design: Implementing different CNN architectures for traffic sign recognition.
* Training and evaluation: Training the models on the dataset, monitoring training progress, and evaluating their performance.
* Transfer Learning: Using a pre-trained VGG16 model for transfer learning and fine-tuning.
* Data Augmentation: Applying data augmentation techniques to improve model generalization.
* Conclusions: Summarizing the findings and insights from the experiments.

## Dependencies
The notebook requires the following libraries to be installed:
* TensorFlow (2.x)
* Keras
* numpy
* pandas
* PIL (Pillow)
* sklearn
* matplotlib

You can install these libraries using the following command:
```
pip install tensorflow keras numpy pandas pillow scikit-learn matplotlib
```

## Dataset
The notebook uses the Traffic Sign Recognition Benchmark (TSRB) dataset for training and evaluation. The dataset includes a collection of traffic sign images with corresponding labels.

Please note that the notebook assumes the dataset is stored in a directory named Train in the same directory as the notebook. Make sure to adjust the path if your dataset is located elsewhere.

## Running the Notebook
To explore the notebook, open it using Jupyter Notebook or JupyterLab and execute the cells step by step. The notebook is structured in a way that guides you through the process of implementing the CNN architectures, training the models, and analyzing the results.

## Conclusions
The experiments and findings in the notebook suggest that the architecture choice has a significant impact on the performance of traffic sign recognition models. Transfer learning and data augmentation did not show substantial improvements for this dataset.