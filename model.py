# model.py - Machine learning model for the Udacity Behavioral Cloning Project
# (c) Jon Wallace 2018 - All rights reserved

from keras.layers import Dense, Conv2D, Dropout, Flatten, Lambda, InputLayer, Cropping2D
from keras.models import Sequential
from keras.utils import Sequence, plot_model
from sklearn.utils import shuffle
import numpy as np
import cv2
from math import ceil
import random

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

def load_batch(img_paths):
    return np.stack([cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB) for img in img_paths])

class SampleImages(Sequence):
    def __init__(self, center_image_paths, left_image_paths, right_image_paths, steering_angles, batch_size):
        self.center_image_paths = center_image_paths
        self.left_image_paths = left_image_paths
        self.right_image_paths = right_image_paths
        self.steering_angles = steering_angles
        self.batch_size = batch_size
        
    def __len__(self):
        return ceil(len(self.center_image_paths) / self.batch_size)
    
    def __getitem__(self, i):
        x = load_batch(self.center_image_paths[i*self.batch_size : (i+1)*self.batch_size])
        y = self.steering_angles[i*self.batch_size : (i+1)*self.batch_size]
        return (x, y)
    
    def on_epoch_end(self):
        self.center_image_paths, self.steering_angles = shuffle(self.center_image_paths, self.steering_angles)
        
def load_data(record_csv, validation_split):
    import csv
    rows = list()
    with open(record_csv, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append(row)
    random.shuffle(rows)
    cutoff = ceil(len(rows) * validation_split)
    center_image_paths = [row[0] for row in rows if len(row) > 0]
    left_image_paths = [row[1] for row in rows if len(row) > 1]
    right_image_paths = [row[2] for row in rows if len(row) > 2]
    steering_angles = [row[3] for row in rows if len(row) > 3]
    training_set = SampleImages(center_image_paths[cutoff:],
                                left_image_paths[cutoff:],
                                right_image_paths[cutoff:],
                                steering_angles[cutoff:],
                                batch_size = 32)

    return training_set, (load_batch(center_image_paths[:cutoff]), steering_angles[:cutoff]) 

training_gen, validation_data = load_data(r'/home/workspace/data/driving_log.csv', 0.2)
print(len(training_gen), len(training_gen.center_image_paths), len(validation_data[0]))
model.compile('adam', 'mse', ['accuracy'])
model.fit_generator(training_gen, validation_data=validation_data, verbose=1, epochs=5, steps_per_epoch=len(training_gen))
model.save(r'model.h5')
