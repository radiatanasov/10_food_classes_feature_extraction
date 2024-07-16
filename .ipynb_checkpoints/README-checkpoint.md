# 10 Food Classes Feature Extraction Project

This project uses feature extraction techniques to classify images into 10 different food categories using 10% of the entire dataset. We leverage two state-of-the-art computer vision model architectures, ResNetV2 and EfficientNet, using TensorFlow Hub for this task.

## Contents

- [Introduction](#introduction)
- [Dataset Information](#dataset-information)
- [Libraries Used](#libraries-used)
- [Data Preparation](#data-preparation)
- [Models and Techniques](#models-and-techniques)
  - [ResNetV2](#resnetv2)
  - [EfficientNet](#efficientnet)
- [Performance Evaluation](#performance-evaluation)
- [Contact](#contact)

## Introduction

The goal of this project is to build a model that can classify images into 10 different food categories using a subset of the dataset, specifically 10% of the total data. We explore two powerful Convolutional Neural Network (CNN) architectures, ResNetV2 and EfficientNet, to perform feature extraction and improve classification accuracy.

## Dataset Information

The dataset is organized into training and testing sets, each containing images of 10 food categories:

- **10_food_classes_10_percent**
  - **test**
    - **chicken_curry**: 250 images
    - **chicken_wings**: 250 images
    - **fried_rice**: 250 images
    - **grilled_salmon**: 250 images
    - **hamburger**: 250 images
    - **ice_cream**: 250 images
    - **pizza**: 250 images
    - **ramen**: 250 images
    - **steak**: 250 images
    - **sushi**: 250 images
  - **train**
    - **chicken_curry**: 75 images
    - **chicken_wings**: 75 images
    - **fried_rice**: 75 images
    - **grilled_salmon**: 75 images
    - **hamburger**: 75 images
    - **ice_cream**: 75 images
    - **pizza**: 75 images
    - **ramen**: 75 images
    - **steak**: 75 images
    - **sushi**: 75 images

## Libraries Used

- tensorflow
- tensorflow_hub
- matplotlib
- numpy
- pandas
- sklearn
- os

## Data Preparation

To prepare the data, the following steps were taken:

### Data Loading and Normalization

The `ImageDataGenerator` class from `tensorflow.keras.preprocessing.image` is used to load and normalize the image data.

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SHAPE = (224, 224)
BATCH_SIZE = 32

train_dir = "10_food_classes_10_percent/train/"
test_dir = "10_food_classes_10_percent/test/"

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

print("Train images:")
train_data = train_datagen.flow_from_directory(train_dir,
                                               target_size=IMG_SHAPE,
                                               batch_size=BATCH_SIZE,
                                               class_mode="categorical")

print("Test images:")
test_data = test_datagen.flow_from_directory(test_dir,
                                             target_size=IMG_SHAPE,
                                             batch_size=BATCH_SIZE,
                                             class_mode="categorical")
```

## Models and Techniques:

We explored two different models for feature extraction: ResNetV2 and EfficientNet.

### ResNetV2

ResNetV2 is a state-of-the-art computer vision model architecture known for its deep residual learning capabilities.

```python
import tensorflow_hub as hub

resnet_url = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5"

resnet_model = tf.keras.Sequential([
    hub.KerasLayer(resnet_url, trainable=False),  # Use ResNet model from TensorFlow Hub
    tf.keras.layers.Dense(train_data.num_classes, activation='softmax')
])

resnet_model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

resnet_model.fit(train_data, epochs=5, validation_data=test_data)
```

### EfficientNet

EfficientNet is another state-of-the-art computer vision architecture that is known for balancing accuracy and efficiency.

```python
efficientnet_url = "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1"

efficientnet_model = tf.keras.Sequential([
    hub.KerasLayer(efficientnet_url, trainable=False),  # Use EfficientNet model from TensorFlow Hub
    tf.keras.layers.Dense(train_data.num_classes, activation='softmax')
])

efficientnet_model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

efficientnet_model.fit(train_data, epochs=5, validation_data=test_data)
```

## Performance Evaluation:

The performance of both models was evaluated using accuracy and loss metrics on the test dataset.

#### ResNetV2 Results

Epoch 5/5
Train Accuracy: 91.47%
Validation Accuracy: 77.48%
Train Loss: 0.3770
Validation Loss: 0.6717

#### EfficientNet Results

Epoch 5/5
Train Accuracy: 83.47%
Validation Accuracy: 75.40%
Train Loss: 0.5535
Validation Loss: 0.7326

# Contact

If you have any questions or suggestions, you can contact me at radi2035@gmail.com.
