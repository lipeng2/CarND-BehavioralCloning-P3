import cv2
import numpy as np

# augment dataset, flip images and -1*measurements
def data_augment(x, y):
    x_aug, y_aug = [], []
    for img, measurement in zip(x, y):
        x_aug.append(img)
        y_aug.append(measurement)
        x_aug.append(cv2.flip(img,1))
        y_aug.append(measurement*-1.0)

    return np.array(x_aug), np.array(y_aug)
