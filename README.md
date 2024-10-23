# tfds-malaria - a Supervised Machine Learning Project

## Specification
A binary classification system detecting malaria parazite infection in blood cell thin smear images.

- Image features: a cropped image depicting a thin smear blood cell
- Classifications: _parasitized_ or _unifected_

## Data sources
The [Tensorflow Malaria dataset](https://www.tensorflow.org/datasets/catalog/malaria) will be used for training and validation.

> The Malaria dataset contains a total of 27,558 cell images with equal instances of parasitized and uninfected cells from the thin blood smear slide images of segmented cells.
>
> &mdash; <cite>https://www.tensorflow.org/datasets/catalog/malaria</cite>

### Relevance
This particular dataset (and other similiar) are exhaustively examined in the ML context and several models are published.

Some notable references are
- [Pre-trained convolutional neural networks as feature extractors toward improved malaria parasite detection in thin blood smear images](https://peerj.com/articles/4568/)
- [Efficient deep learning-based approach for malaria detection using red blood cell smears](https://www.nature.com/articles/s41598-024-63831-0)

In such distinguished company, this work is not expected to be a competitor regardning quality and thoroughness... 

## Artifacts
- A commandline utility for binary classification of thin smear blood cell images
- The [explorative work and resoning]((https://github.com/jlarsson/tfds-malaria/blob/main/malaria.ipynb)) will be published online as a python notebook 

## Features
- Data analysis
    - investigate overall data quality and completeness
    - investigate possible outliers or systematic errors/biases in the dataset
        - devise a strategy that avoids unnescessary overfit
        - devise training metrics
    - recommend normalizations (of images)
- Model training and validation
    - one or more convoluted neural networks
        - different networks may differ in
            - speed, accuracy or preprocessing needs
            - architectural choices such as ad hoc models or hybrid models with pretrained layers (VGG16, VGG19)
    - investigate model designs using k-fold cross validation
    - showcase model performance 
- Preprocessors
    - if images needs conversion to be accepted by the model
- Prediction
    - a predictor (or multiple) that normalizes input images and classifies using a pretrained model


