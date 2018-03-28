import csv
import cv2
import numpy as np

def get_lines(dir):
    lines = []
    with open(f'{dir}driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    return lines

def get_data(dir, addition=False, adjustment=0.2):
    images = []
    measurements = []
    for line in get_lines(dir):
        file = line[0].split('\\')[-1]
        img = cv2.imread(f'{dir}IMG/{file}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        measurements.append(float(line[3]))
        if addition:
            for i in range(1,3):
                file = line[i].split('\\')[-1]
                img = cv2.imread(f'{dir}IMG/{file}')
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                measurement = float(line[3])
                if i == 1: measurement += adjustment
                if i == 2: measurement -= adjustment
                measurements.append(measurement)

    X_train = np.array(images)
    y_train = np.array(measurements)

    return X_train, y_train
