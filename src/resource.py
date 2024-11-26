import os
import requests
import tensorflow as tf

def try_get_binary_content_from_file(path: str):
    '''
    Load a file from disk
    
    Returns: 
        - binary content if file exists, None otherwise
    Raises:
        - all errors
    '''
    if os.path.exists(path):
        with open(path, 'rb') as file:
            return file.read()
    return None

def try_get_binary_content_from_url(url: str):
    '''
    Load binary content from url
    Returns: 
        - binary content if found
        - None if 404 NOT_FOUND intenet resource
    Raises:
        - all errors except 404 NOT_FOUND
    '''
    if url.startswith(('http://', 'https://')):
        response = requests.get(url)
        if response.status_code == 404:
            return None
        response.raise_for_status()  # Fail of other than 2XX OK
        return response.content
    return None
