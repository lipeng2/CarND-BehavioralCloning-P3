import retrive_data as rd
import process_data as pro
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D, Dropout
from keras import optimizers
import matplotlib.pyplot as plt

# directory of training dataset
dir = 'advance_train/'

# obtain preprocess training data
x,y = rd.get_data(dir, addition=True, adjustment=0.2)
x_aug, y_aug = pro.data_augment(x,y)

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
if len(dir) > 7:
    model.add(Dropout(0.6))
# fc with 100 ouput units
model.add(Dense(100))
# fc with 50 output units
model.add(Dense(50))
# fc with 10 output units
model.add(Dense(10))
# fc with 1 output unit
model.add(Dense(1))

# compile model using adam optimizer with learning rate of 0.0001 wiht no decay
adam = optimizers.Adam(lr=0.0001, decay=1e-6)
model.compile(loss='mean_squared_error', optimizer=adam)
model.fit(x_aug, y_aug, epochs=10, validation_split=0.2, shuffle=True)

# save model
model.save('advance_model.h5')
