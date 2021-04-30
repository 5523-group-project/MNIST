import os

import cv2
import numpy as np

IMAGE_SIZE = 64


def get_files(directory: str):
    f = []
    for (dirpath, dirnames, filenames) in os.walk(directory):
        for file_name in filenames:
            file = os.path.join(dirpath, file_name)
            f.append(file)
    return f


def transform_image(image: np.ndarray):

    # threshold so it is 255 for the writing and 0 for all else
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)

    # max_len = max(w, h)

    # crop out black edges
    image = image[y:y+h, x:x+w]

    # resize
    image = cv2.resize(image, (8, 8), interpolation=cv2.INTER_CUBIC)

    norm_div = np.max(image) - 0.5

    mult  = 15 / norm_div
    image = np.round(image * mult)

    new_max = np.max(image)
    new_min = np.min(image)

    image = image.reshape(-1)
    assert new_max == 15.0
    assert new_min == 0.0
    assert image.shape[0] == IMAGE_SIZE

    return image


def process(directory: str) -> np.ndarray:
    files = get_files(directory)
    images = [cv2.imread(file, flags=cv2.IMREAD_GRAYSCALE) for file in files]
    image_count = len(images)

    arr = np.empty((image_count, IMAGE_SIZE))

    for i, image in enumerate(images):
        arr[i] = transform_image(image)

    return arr
