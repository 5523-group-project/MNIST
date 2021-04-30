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

    resulting_images = []

    bounding_rectangles = [cv2.boundingRect(cnt) for cnt in contours]

    # sort by x
    list.sort(bounding_rectangles, key=lambda x: x[0])

    for bound in bounding_rectangles:
        x, y, w, h = bound
        sub_image = image[y:y + h, x:x + w]

        # resize
        sub_image = cv2.resize(sub_image, (8, 8), interpolation=cv2.INTER_CUBIC)

        norm_div = np.max(sub_image) - 0.5

        mult = 15 / norm_div
        sub_image = np.round(sub_image * mult)

        new_max = np.max(sub_image)
        new_min = np.min(sub_image)

        sub_image = sub_image.reshape(-1)
        assert new_max == 15.0
        assert new_min == 0.0
        assert sub_image.shape[0] == IMAGE_SIZE
        resulting_images.append(sub_image)

    return resulting_images


def process(directory: str) -> np.ndarray:
    files = get_files(directory)
    images = [cv2.imread(file, flags=cv2.IMREAD_GRAYSCALE) for file in files]

    arr = np.empty((0, IMAGE_SIZE))

    for i, image in enumerate(images):
        for sub_image in transform_image(image):
            arr = np.append(arr, sub_image.reshape(1,-1), axis=0)
    return arr
