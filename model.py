# model.py - Machine learning model for the Udacity Behavioral Cloning Project
# (c) Jon Wallace 2018 - All rights reserved

from keras.layers import Dense, Conv2D, Dropout, Flatten, Lambda, InputLayer, Cropping2D
from keras.models import Sequential
from keras.utils import Sequence, plot_model
import numpy as np

model = Sequential()

model.add(InputLayer(input_shape=(160, 320, 3)))
model.add(Lambda(lambda x:x / 128 - 1))
model.add(Cropping2D(((50, 20), (0, 0))))

model.add(Conv2D(24, 5, strides=(2, 2), activation='relu'))
model.add(Conv2D(36, 5, strides=(2, 2), activation='relu'))
model.add(Conv2D(48, 5, strides=(2, 2), activation='relu'))
model.add(Conv2D(64, 3, activation='relu'))
model.add(Conv2D(64, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

def data_loader(record_csv):
    import csv
    with open(record_csv, 'rb') as f:
        reader = csv.reader(f)
        # do something
    while True:
        yield (np.zeros((1, 160, 320, 3)), np.zeros((1, 1)))

model.compile('adam', 'mse', ['accuracy'])
model.fit_generator(data_loader(r'test.csv'), steps_per_epoch=1, epochs=1)
model.save(r'model.h5')

print(model.summary())
plot_model(model, show_shapes=True)