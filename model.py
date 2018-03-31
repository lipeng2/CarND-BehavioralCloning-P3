import retrive_data as rd
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D, Dropout
from keras import optimizers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

# directory of training dataset
dir = 'train/'
samples = rd.get_lines(dir)

# obtain training and validation sets
train_samples, val_samples = train_test_split(samples, test_size=0.2)

# parameters
keep_prob = 0.3 # keep_prob for dropout layer
epochs = 5 # epochs for training
batch_size = 12 # batch_size
learning_rate = 0.0001 # learning rate of adam optimizer

# create data generators
train_generator = rd.generator(train_samples, batch_size, addition=False)
val_generator = rd.generator(val_samples, batch_size)

# model
model = Sequential()
# normalize data
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
# cropping out the top and bottom portion of image frames
model.add(Cropping2D(cropping=((70,25), (0,0))))
# conv 24 filters 5x5 kernel
model.add(Conv2D(24,5,5,activation='relu'))
# conv 36 filters 5x5 kernel
model.add(Conv2D(36,5,5,activation='relu'))
# conv 48 filters 5x5 kernel
model.add(Conv2D(48,5,5,activation='relu'))
# conv 64 filter 3x3 kernel
model.add(Conv2D(64,3,3,activation='relu'))
# conv 64 filter 3x3 kernel
model.add(Conv2D(64,3,3,activation='relu'))
# fully connected layer
model.add(Flatten())
# fc with 100 ouput units
model.add(Dense(100, activation='relu'))
# fc with 50 output units
model.add(Dense(50, activation='relu'))
# dropout layer with keep_prob
model.add(Dropout(keep_prob))
# fc with 10 output units
model.add(Dense(10, activation='relu'))
# fc with 1 output unit
model.add(Dense(1))

# compile model using adam optimizer with learning rate of 0.0001 wiht no decay
adam = optimizers.Adam(lr=learning_rate, decay=1e-6)
model.compile(loss='mean_squared_error', optimizer=adam)

model.fit_generator(train_generator, steps_per_epoch= len(train_samples)//batch_size, validation_data=val_generator, validation_steps=len(val_samples)//batch_size, epochs=5, shuffle=True, workers=4)

# save model
model.save('model.h5')
