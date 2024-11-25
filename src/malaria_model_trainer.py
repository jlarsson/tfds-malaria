import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Conv2D, Input, MaxPooling2D, Flatten, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, History, ModelCheckpoint, TensorBoard

class MalariaModelTrainer:
    '''
    Class for creating a trained Malaria model

    The notebook malaria.ipynb covers reasoning and much code is adapted from there.
    '''
    verbose: int
    def __init__(self, verbose = 1):
        self.verbose = verbose
    def train_model(self) -> Model:
        AUTOTUNE=tf.data.AUTOTUNE

        def conv(n: int):
            '''
            Utility for setting up convolutional layer
            '''
            return Sequential([
                Conv2D(n, (3, 3), activation="relu"),
                MaxPooling2D((2, 2)),
            ])

        def dense(n):
            '''
            Utility for setting up dense layer
            '''
            return Dense(n, activation="relu")

        def compile_malaria_model(model: Model) -> Model:
            '''
            Compile a Malaria CNN
            '''
            model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
            return model

        # Create a model suited for our dataset and classification
        def create_malaria_model(hidden_layers: list[Model], **kvargs) -> Model:
            '''
            Create a Malaria CNN with given inner layers
            '''
            return compile_malaria_model(Sequential([
                # 121x121 images with 1 color channel
                Input(shape=(121, 121, 1)),
                # Model specific hidden layers
                *hidden_layers,
                # Binary classification
                Dense(1, activation="sigmoid")],
                **kvargs))

        def load_and_normalize():
            '''
            Load the malria dataset and convert images to 121x121 grayscale images
            '''
            def cvt(image, label):
                image = tf.image.rgb_to_grayscale(image)
                image = tf.image.resize(image, [121,121])
                image = image / 255.0
                return image, label
            def normalize(ds):
                return ds.map(cvt, num_parallel_calls=AUTOTUNE).batch(32).prefetch(AUTOTUNE)
            (ds_train, ds_test, ds_validate), ds_info = tfds.load(
                'malaria',
                split=["train[0%:20%]", "train[20%:25%]", "train[25%:]"],
                as_supervised=True,
                with_info=True
            )
            return (normalize(ds_train), normalize(ds_test), normalize(ds_validate)), ds_info

        # Load and normalize the malaria dataset
        # Partition into train, test and validate (unused)
        (ds_train, ds_test, ds_validate), ds_info = load_and_normalize()

        # Create the model
        # For details regardning inner layers, see reasoning in notebook malaria.ipynb
        model = create_malaria_model([
            conv(16),
            conv(32),
            conv(128),
            Flatten(),
            dense(64),
            dense(32),
            dense(16)
        ])

        # Do the actual training
        model.fit(
            ds_train,
            validation_data = ds_test,
            epochs=100,
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True, verbose=self.verbose),
            ],
            verbose=self.verbose)
        
        return model

