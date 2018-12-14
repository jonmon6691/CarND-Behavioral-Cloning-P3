# model.py - Machine learning model for the Udacity Behavioral Cloning Project
# (c) Jon Wallace 2018 - All rights reserved

from keras.layers import Dense, Conv2D, Dropout, Flatten, Lambda, InputLayer, Cropping2D
from keras.utils import Sequence, plot_model
from keras.models import Sequential
from sklearn.utils import shuffle
from math import ceil
import numpy as np
import random
import cv2
import csv

# Define machine learning model to include preprocessing steps
model = Sequential()
# Image from vehicle
model.add(InputLayer(input_shape=(160, 320, 3)))
# Normalization
model.add(Lambda(lambda x:x / 128 - 1))
# Crop away parts of the image that don't generally contain lane lines such as above the horizon and below the hood.
model.add(Cropping2D(((50, 20), (0, 0))))
# CNN Model from NVIDIA end-to-end learning for self driving cars paper (Bojarski et. al.)
# https://arxiv.org/pdf/1604.07316v1.pdf
model.add(Conv2D(24, 5, strides=(2, 2), activation='relu'))
model.add(Conv2D(36, 5, strides=(2, 2), activation='relu'))
model.add(Conv2D(48, 5, strides=(2, 2), activation='relu'))
model.add(Conv2D(64, 3, activation='relu'))
model.add(Conv2D(64, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1)) # Steering output

def load_batch(img_paths):
    # Takes a list of file path strings and loads them into a numpy array ready for training
    return np.stack([cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB) for img in img_paths])

class SampleImages(Sequence):
    # Batching class used by model.fit_generator(), it loads the training images in batches
    def __init__(self, center_image_paths, left_image_paths, right_image_paths, steering_angles, batch_size):
        self.center_image_paths = center_image_paths
        self.left_image_paths = left_image_paths
        self.right_image_paths = right_image_paths
        self.steering_angles = steering_angles
        self.batch_size = batch_size
        
    def __len__(self):
        return ceil(len(self.center_image_paths) / self.batch_size)
    
    def __getitem__(self, i):
        left_x = load_batch(self.left_image_paths[i*self.batch_size : (i+1)*self.batch_size])
        right_x = load_batch(self.right_image_paths[i*self.batch_size : (i+1)*self.batch_size])
        x = load_batch(self.center_image_paths[i*self.batch_size : (i+1)*self.batch_size])
        y = self.steering_angles[i*self.batch_size : (i+1)*self.batch_size]
        left_y = y + 0.25
        right_y = y - 0.25
        return (np.stack(x, left_x, right_x), np.stack(y, left_y, right_y))
    
    def on_epoch_end(self):
        # Shuffle the data between epochs
        self.center_image_paths, self.steering_angles = shuffle(self.center_image_paths, self.steering_angles)
        
def load_data(record_csv, validation_split):
    # Loads the driving_log.csv file and creates the test-validation data structures

    rows = list()
    with open(record_csv, 'r') as f:
        reader = csv.reader(f)
        rows = [row for row in reader if len(row) > 3]
    
    # Shuffle the deck 
    random.shuffle(rows)
    
    # Separate the CSV rows
    center_image_paths = [row[0] for row in rows]
    left_image_paths = [row[1] for row in rows]
    right_image_paths = [row[2] for row in rows]
    steering_angles = [row[3] for row in rows]
    
    # Cut the deck according to the validation split 
    cutoff = ceil(len(rows) * validation_split)

    # Use part to feed the training batch generator
    training_set = SampleImages(center_image_paths[cutoff:],
                                left_image_paths[cutoff:],
                                right_image_paths[cutoff:],
                                steering_angles[cutoff:],
                                batch_size = 32)

    # Use the rest for validation
    validation_data = (load_batch(center_image_paths[:cutoff]), steering_angles[:cutoff])
    
    return training_set, validation_data 

if __name__ == "__main__":
    # Load and prepare the data for learning
    training_gen, validation_data = load_data(r'/home/workspace/data/driving_log.csv', 0.2)
    print("Training model using:\n\t%d batches,\n\t%d total training images,\n\t%d total validation images\n" % (len(training_gen), len(training_gen.center_image_paths), len(validation_data[0])))

    # mse optimizer used because we are trying to fit a single value, the steering angle, to the training data
    model.compile('adam', 'mse', ['accuracy'])
    model.fit_generator(training_gen, validation_data=validation_data, verbose=1, epochs=5, steps_per_epoch=len(training_gen))
    model.save(r'model.h5')
