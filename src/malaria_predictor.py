import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import Model
from .malaria_model_trainer import MalariaModelTrainer

class MalariaPredictor:
    '''
    Class for prediction of Malaria images
    Trains model on demand
    Loads and saves model to disk
    '''
    verbose: int
    model_path: str
    model: Model
    def __init__(self, model_path: str, verbose=1):
        self.verbose = verbose
        self.model = None
        self.model_path = model_path

    def predict(self, image: tf.Tensor):
        '''
        Given an image, correctly coded as a tensor, return (sigmoid) prediction
        '''
        model = self._ensure_model()
        dataset = tf.data.Dataset.from_tensors([image])
        p = model.predict(dataset, verbose=self.verbose)
        return p[0][0]

    def _ensure_model(self) -> Model:
        return self.model or self._try_load_model() or self._try_train_model()

    def _try_load_model(self) -> Model:
        '''
        Return previsouly saved model, if any
        '''
        if os.path.exists(self.model_path):
            return load_model(self.model_path)
        return None
        
    def _try_train_model(self) -> Model:
        '''
        Train model and save weights etc
        '''
        model = MalariaModelTrainer(verbose=self.verbose).train_model()
        model.save(self.model_path)
        return model
