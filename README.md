# tfds-malaria - a Supervised Machine Learning Project

## Usage
```
pip install -r requirements.txt
python malaria.py --src <path or url to image> [--silent]
```

The main program [malaria.py](malaria.py) attempts to load image denoted by commandline parameter __--src__ and performs classification into __Parasitized__ or __Unifected__.

I the file __model.keras__ doent exists, a new model is trained and saved to that file for future usage.
For details about the model and the underlying reasoning, please check out [malaria.ipynb](malaria.ipynb).

### Sample usage
[https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria) can be used as a source for testing the tool.

```
python malaria.py --src 'https://storage.googleapis.com/kagglesdsdata/datasets/87153/200743/cell_images/Parasitized/C100P61ThinF_IMG_20150918_144104_cell_162.png?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20241124%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20241124T110109Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=1ccf49edfec200b9c7cf540979306c622ef311ebb657000d963a42759ffe76582b7e11684736b2338ad6520dfcc9f3bd506febd60d4ad32a8dcf02f57a8bcc93f522d98334da893d7f1eae5d1814a632d4ca9a6cdc13999a95e2574da5d7fc355ea4ea27c3a9b00e16b1f5bec48bfee8eaaf38bf10e7c3b8aba322ab5cdca182d1d184e6cba386552e8641403d7898f8aaf8a3ca4e71cb677583b9304cf109cab75ed82893f049f62bcc99892e045acf6b5fc026f3b7c563e2558acdd49ca6e2afd94a138cff5cfc335b1e2be4c8727fe326f706aeb613fb6124d7d364d78c77d6783fdad52f92c927f770bf55cd84f03b2902448bc3ce31ca042283fe67fdaa'

python malaria.py --src 'https://storage.googleapis.com/kagglesdsdata/datasets/87153/200743/cell_images/Uninfected/C100P61ThinF_IMG_20150918_144104_cell_128.png?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20241125%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20241125T154810Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=264457827c4259123c6ca34e5e567fc95f8cdc434cfe7c5e4e613e1345216c4c7dd965c83fbfae7a2b937465d3d4cf876be0e6d81215c0238314bee21d9ecf0499397e28faf0784b90f17ddfed517f6b0fda08f505a0c1c431dd8e455f54359bdd5d0cc2d9a52ef138602e5b4eda80bac7a5884d6b98ecc9d9f556311cba319061317a5464c3fb9513167fdbf34e529361ff7d341ac1e94ee200143e9d4b149c06558ee6d03f14ff88510c708965824d49afb829a7f7ea62c3e2c6eff75dbe41b2ba5fd8849ad87e64540da6511b2d3441671a015368299fad2ee62570afe144c7cc2f6e8f4f1cc534a7b048633d4be8f8e88a5aaf1662bb15e9680d199b490f'
```

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

In such distinguished company, this work is not expected to be a competitor regarding quality and thoroughness... 

## Artifacts
- A commandline utility for binary classification of thin smear blood cell images
- The [explorative work and reasoning](https://github.com/jlarsson/tfds-malaria/blob/main/malaria.ipynb) will be published online as a python notebook 

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
    - conversion of images in "natural" format (jpg, png etc), to the formats required by the models
- Prediction
    - a predictor (or multiple) that normalizes input images and classifies using a pretrained model
