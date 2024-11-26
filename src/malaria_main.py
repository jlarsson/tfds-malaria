from .resource import try_get_binary_content_from_file, try_get_binary_content_from_url
from .malaria_predictor import MalariaPredictor

def malaria_main(resource: str, verbose: int):
    '''
    Main function in malaria detection
    - Load resource from disk or internet
    - decode to image tensor
    - predict
    '''
    # Load input data
    data = try_get_binary_content_from_file(resource) or try_get_binary_content_from_url(resource)
    if data is None:
        raise Exception(f'Image resource "{resource}" not found')
    
    # get input image from file or url and convert to 121x121 grayscale
    image = MalariaPredictor.convert_binary_to_image_tensor(data)

    # predict
    prediction = MalariaPredictor('malaria-model.keras', verbose=verbose).predict(image)
    if verbose:
        print(f'prediction={prediction} (0 = parasitized, 1 = uninfected)')
    print('Parasitized' if round(prediction) == 0 else 'Uninfected')
