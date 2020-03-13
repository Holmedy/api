import flask
from flask import Flask
from flask import request
import json
import urllib
import requests
import numpy as np
from pathlib import Path, PurePath
from typing import Dict, List, Optional, Union
from image_utils import load_image, preprocess_image
from logger import return_logger


app = Flask(__name__)


class CNN:
    """
    Find duplicates using CNN and/or generate CNN encodings given a single image or a directory of images.
    The module can be used for 2 purposes: Encoding generation and duplicate detection.
    - Encodings generation:
    To propagate an image through a Convolutional Neural Network architecture and generate encodings. The generated
    encodings can be used at a later time for deduplication. Using the method 'encode_image', the CNN encodings for a
    single image can be obtained while the 'encode_images' method can be used to get encodings for all images in a
    directory.
    - Duplicate detection:
    Find duplicates either using the encoding mapping generated previously using 'encode_images' or using a Path to the
    directory that contains the images that need to be deduplicated. 'find_duplciates' and 'find_duplicates_to_remove'
    methods are provided to accomplish these tasks.
    """

    def __init__(self, verbose: bool = True) -> None:
        """
        Initialize a keras MobileNet model that is sliced at the last convolutional layer.
        Set the batch size for keras generators to be 64 samples. Set the input image size to (224, 224) for providing
        as input to MobileNet model.
        Args:
            verbose: Display progress bar if True else disable it. Default value is True.
        """
        from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
        from data_generator import DataGenerator

        self.MobileNet = MobileNet
        self.preprocess_input = preprocess_input
        self.DataGenerator = DataGenerator

        self.target_size = (224, 224)
        self.batch_size = 64
        self.logger = return_logger(__name__)
        self._build_model()
        self.verbose = 1 if verbose is True else 0

    def _build_model(self):
        """
        Build MobileNet model sliced at the last convolutional layer with global average pooling added.
        """
        self.model = self.MobileNet(
            input_shape=(224, 224, 3), include_top=False, pooling="avg"
        )

        self.logger.info(
            "Initialized: MobileNet pretrained on ImageNet dataset sliced at last conv layer and added "
            "GlobalAveragePooling"
        )

    def encode_image(
        self,
        image_file: Optional[Union[PurePath, str]] = None,
        image_array: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate CNN encoding for a single image.
        Args:
            image_file: Path to the image file.
            image_array: Optional, used instead of image_file. Image typecast to numpy array.
        Returns:
            encoding: Encodings for the image in the form of numpy array.
        Example:
        ```
        from imagededup.methods import CNN
        myencoder = CNN()
        encoding = myencoder.encode_image(image_file='path/to/image.jpg')
        OR
        encoding = myencoder.encode_image(image_array=<numpy array of image>)
        ```
        """
        if isinstance(image_file, str):
            image_file = Path(image_file)

        if isinstance(image_file, (PurePath, urllib.request.http.client.HTTPResponse)):

            if isinstance(image_file, PurePath) and not image_file.is_file():
                raise ValueError(
                    "Please provide either image file path or image array!"
                )

            image_pp = load_image(
                image_file=image_file, target_size=self.target_size, grayscale=False
            )

        elif isinstance(image_array, np.ndarray):
            image_pp = preprocess_image(
                image=image_array, target_size=self.target_size, grayscale=False
            )
        else:
            raise ValueError("Please provide either image file path or image array!")

        return (
            self._get_cnn_features_single(image_pp)
            if isinstance(image_pp, np.ndarray)
            else None
        )

    def _get_cnn_features_single(self, image_array: np.ndarray) -> np.ndarray:
        """
        Generate CNN encodings for a single image.
        Args:
            image_array: Image typecast to numpy array.
        Returns:
            Encodings for the image in the form of numpy array.
        """
        image_pp = self.preprocess_input(image_array)
        image_pp = np.array(image_pp)[np.newaxis, :]
        return self.model.predict(image_pp)


def computeFeaturesVector(imageUrl):
    # ... Process image
    myencoder = CNN()
    res = urllib.request.Request(imageUrl, headers={"User-Agent": "Mozilla/5.0"})
    response = urllib.request.urlopen(res)
    vector = myencoder.encode_image(image_file=response)
    vector = vector.tolist()
    return vector


def checkAuthorization(authHeader):
    print(authHeader)
    if authHeader == "Bearer " + "hubstairs-login":
        return True
    else:
        return False


@app.route("/rest-test", methods=["GET"])
def restTest():

    authHeader = request.headers.get("Authorization")
    tokenCheck = checkAuthorization(authHeader)

    if tokenCheck == True:
        status_code = flask.Response(status=200)
        return status_code
    else:
        status_code = flask.Response(status=401)
        return status_code


@app.route("/compute-vector", methods=["POST"])
def processPostRequest():

    authHeader = request.headers.get("Authorization")
    tokenCheck = checkAuthorization(authHeader)

    if tokenCheck == True:
        payload = request.get_json()
        imageUrl = payload["imageUrl"]
        resultFeatureVector = computeFeaturesVector(imageUrl)
        resultPayload = json.dumps({"featuresVector": resultFeatureVector})
        return resultPayload
    else:
        return flask.Response(status=401)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

