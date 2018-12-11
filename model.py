# model.py - Machine learning model for the Udacity Behavioral Cloning Project
# (c) Jon Wallace 2018 - All rights reserved

from keras.layers import Dense, Conv2D, Dropout, Flatten, Lambda, InputLayer, Cropping2D
from keras.models import Sequential
from keras.utils import Sequence, plot_model
import numpy as np
import cv2

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

def data_loader(record_csv, batch_size=32):
    import csv
    rows = list()
    with open(record_csv, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append(row)

    center_image_paths = [row[0] for row in rows if len(row) > 0]
    left_image_paths = [row[1] for row in rows if len(row) > 1]
    right_image_paths = [row[2] for row in rows if len(row) > 2]
    steering_angles = [row[3] for row in rows if len(row) > 3]
    
    while True:
        for offset in range(0, len(rows), batch_size):
            x = np.stack([cv2.imread(img) for img in center_image_paths[offset:offset+batch_size]])
            print(x.shape)
            y = steering_angles[offset:offset+batch_size]
            yield (x, y)

model.compile('adam', 'mse', ['accuracy'])
model.fit_generator(data_loader(r'examples/driving_log.csv'), steps_per_epoch=1, nb_epochs=1)
model.save(r'model.h5')
