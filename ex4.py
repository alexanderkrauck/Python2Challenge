"""
Author: Alexander Krauck
Matr.Nr.: k11904235
Exercise 4
"""

import numpy as np

def ex4(image_array: np.ndarray, border_x: tuple, border_y: tuple):
    
    if not isinstance(image_array, np.ndarray):
        raise NotImplementedError
    
    if len(image_array.shape) != 2:
        raise NotImplementedError

    #Here try converting to int and if it does not work throw Value Error
    try:
        x1, x2 = border_x
        x1, x2 = int(x1), int(x2)

        y1, y2 = border_y
        y1, y2 = int(y1), int(y2)
    except BaseException:
        raise ValueError

    if x1 < 1 or x2 < 1 or y1 < 1 or y2 < 1:
        raise ValueError

    #transform to coordinates
    x2 = image_array.shape[0] - x2
    y2 = image_array.shape[1] - y2

    if x2 - x1 < 16 or y2 - y1 < 16:
        raise ValueError



    #Target Array
    mask = np.ones_like(image_array)
    mask[x1:x2, y1:y2] = 0
    mask = mask.astype(bool)
    target_array = image_array[mask]
    target_array = target_array.astype(image_array.dtype)

    #Input Array
    input_array = np.zeros_like(image_array, dtype=image_array.dtype)
    input_array[x1:x2, y1:y2] = image_array[x1:x2, y1:y2]

    #Known Array
    known_array = np.zeros_like(image_array, dtype=image_array.dtype)
    known_array[x1:x2, y1:y2] = 1


    return (input_array, known_array, target_array)

