import csv
import cv2
import numpy as np
from sklearn.utils import shuffle

def get_lines(dir):
    '''
    Input:
    dir -- name of the dictory where the dataset is stored

    Return:
    lines -- list of filenames of images and values of steering angles
    '''
    lines = []
    with open(f'{dir}driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    return lines

# helper function
def generate_data(batch_samples, addition=False, adjustment=0.2):
    images, angles = [], []
    for batch_sample in batch_samples:
        num_of_cameras = 3 if addition else 1
        for i in range(num_of_cameras):
            filename = batch_sample[i]
            img = cv2.imread(filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            angle = float(batch_sample[3])
            if i ==1: angle += adjustment
            if i ==2: angle -= adjustment
            images.append(img)
            images.append(cv2.flip(img,1))
            angles.append(angle)
            angles.append(-1.0*angle)
    return np.array(images), np.array(angles)

# data generator
def generator(samples, batch_size=32, addition=False, adjustment=0.2):
    '''
    Input:
    samples -- Dataset from which data generated
    batch_size -- size of a single batch of samples
    addition -- boolean value. Default value is False. When equal true, generators use data obtained from all three cameras
    adjustment -- The value add or substracted from angles when using data obtained from all three cameras

    Return:
    Images -- list of images data
    angles -- list of steering angles
    '''
    n = len(samples)
    while True:
        samples = shuffle(samples)
        for offset in range(0, n, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            yield generate_data(batch_samples, addition, adjustment)
