import os
import requests
import tensorflow as tf

class LoadResource:
    def load(self, resource):
        return None

class LoadResourceFromFile(LoadResource):
    def __init__(self, inner: LoadResource = None):
        self.inner = inner
        '''
        Load a file from disk
        
        Returns: 
            - binary content if file exists, None otherwise
        Raises:
            - all errors
        '''
    def load(self, resource):
        if os.path.exists(resource):
            with open(resource, 'rb') as file:
                return file.read()
        return self.inner.load(resource) if self.inner else None
    
class LoadResourceFromUrl(LoadResource):
    def __init__(self, inner: LoadResource = None):
        self.inner = inner

    def load(self, resource):
        '''
        Load binary content from url
        Returns: 
            - binary content if found
            - None if 404 NOT_FOUND intenet resource
        Raises:
            - all errors except 404 NOT_FOUND
        '''
        if resource.startswith(('http://', 'https://')):
            response = requests.get(resource)
            if response.status_code == 404:
                return None
            response.raise_for_status()  # Fail of other than 2XX OK
            return response.content
        return self.inner.load(resource) if self.inner else None

class ConvertImageResource(LoadResource):
    def __init__(self, inner: LoadResource = None):
        self.inner = inner


    def load(self, resource):
        # get actual bytes
        image_bytes = self.inner.load(resource)
        if image_bytes is not None:
            # decode into supported image format
            image = tf.image.decode_image(image_bytes) if image_bytes else None
            # convert to grayscale
            image = tf.image.rgb_to_grayscale(image)
            # resize to our trained resolution
            image = tf.image.resize(image, [121,121])
            # normalize
            return image / 255.0
        return None
