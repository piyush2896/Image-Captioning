import tensorflow as tf
import tensorflow_hub as hub

def mobilenet_module():
    """Get a mobile net v1 module with.
    Returns:
        module: mobile net tf-hub module
        height: of input image
        width: of input image
        num_features: number of output features 
    """
    module = hub.Module(
        "https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/feature_vector/1", trainable=False)
    height, width = hub.get_expected_image_size(module)
    num_features = 1024
    return module, height, width, num_features
