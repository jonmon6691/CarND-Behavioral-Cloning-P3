# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

![](cover.png)

Overview
---
This repository contains files for the Behavioral Cloning Project.

In this project, deep neural networks and convolutional neural networks clone driving behavior. The model is trained, validated and tested using Keras. The model outputs a steering angle to an autonomous vehicle.

Udacity has provided a simulator where one can steer a car around a track for data collection. Image data and steering angles were used to train a neural network. This model was then used to drive the car autonomously around the track.

To meet specifications, the project will require submitting five files: 
* model.py (script used to create and train the model)
* drive.py (script to drive the car)
* model.h5 (a trained Keras model)
* writeup.md
* video.mp4 (a video recording of the vehicle driving autonomously around the track for at least one full lap)

This README file describes how to output the video in the "Details About Files In This Directory" section.

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

## Details About Files In This Directory


The model can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.


The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.

# Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to transform the vehicle state at an instant in time into a steering angle for that instant. The vehicle state is captured as a 2D color image taken from a camera placed on the vehicle center looking forward. When working with image data as an input, a convolutional neural network is a good choice. CNN's are robust against translational variation and other aspects of image data and are good for extracting high-level features like lane lines. Model architecture is an inexact science and there are near infinite CNN architectures that would solve the problem presented in this project. In the face of that indeterminance, I chose to start with an architecture designed for the purpose of vehicle control. The NVIDIA vehicle control CNN turned out to be more than sufficient for the job.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. My model performed perfectly literally the first time that I tried it. I did not need to collect more training data, or adjust any hyper-parameters. I was completely shocked and spent 10 minutes making sure I wasn't just playing back the training data or something.

#### 2. Final Model Architecture

The final model architecture (model.py lines 15-33) consisted of a convolution neural network with the following layers and layer sizes :

![](model.png)

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track one using center lane driving. Here is an example image of center lane driving:

![](examples\input_example.jpg)

I then recorded the vehicle going around the track three times in the reverse direction.

After training with that data, my model was preforming perfectly on track one and I did not need to do any data augmentation like reversing or collecting recovery driving data.

After the collection process, I had about 20k data points. I then preprocessed this data by converting to RGB, normalizing to [-1,1] and cropping the top and bottom off.


I finally randomly shuffled the data set and put 20% of the data into a validation set. I used the fit_generator() method and wrote a class that implemented the Sequence interface provided by Keras. This means that during training, only the images being used in the current batch need to be loaded into memory.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the test performance on the test track. I used an adam optimizer so that manually training the learning rate wasn't necessary.