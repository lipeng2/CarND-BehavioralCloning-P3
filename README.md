# CarND-Behavioral-Cloning-P3

## Installation
To create an environment for this project use the following command:
```
conda env create -f environment-gpu.yml
```
You will also required to download a [simulator](https://github.com/udacity/self-driving-car-sim) created by udacity and Nvidia team for this project. The simulator is used for creating training data and running testing simulation. 

## Overview 
The project is to train a convolutional neural network (CNN) to learn driving in traffic on local roads with or without lane markings. The input is a series of images taken from 3 front facing cameras positioned in the front of a vehicle, and the output of the model is steering command. Sample images obtained from 3 cameras that positioned in different angles(from left to right) shown below. 
<p float="left">
  <img src="https://github.com/lipeng2/CarND-BehavioralCloning-P3/blob/master/train_examples/left_2018_03_29_10_59_08_399.jpg" width="250" />
  <img src="https://github.com/lipeng2/CarND-BehavioralCloning-P3/blob/master/train_examples/center_2018_03_29_10_59_08_399.jpg" width="250" /> 
  <img src="https://github.com/lipeng2/CarND-BehavioralCloning-P3/blob/master/train_examples/right_2018_03_29_10_59_08_399.jpg" width="250" />
</p>

### The steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road

### Model overview
* The trained model implemented in [model.py](https://github.com/lipeng2/CarND-BehavioralCloning-P3/blob/master/model.py) is proposed by the paper [End to End Learning for Self-Driving Cars](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). 
* It consists of 9 layers, including a normalization layer, a cropping layer, 5 convolutional layers with relu activation functions, and 3 fully connected layers. The convolutional layers are designed for feature extraction. 
* The first three convolutional layers use 5x5 kernels and 2x2 strides, and the last two convolutional layers use 3x3 kernels with no stride. 
* Following convolutional layers, 3 fully connected layers and one dropout layer with keep_prob=0.3 are implemented to introduce nonlinearity into the model. 

### Solution Approch
* Initial attemp is to implement the model proposed by Nvidia team, and train the model using adam optimizer. As examining the training loss, and validation loss. Training loss drops quickly under 0.02 after about 4 epochs, while the validation loss starts to increase dramatically. Therefore, we can conclude that our model is overfitting the training data.
* Steps to address overfitting:
  1. Generate more training data to compensate for the complexity of proposed model. 
  2. Employ dropout techniques
  
### Video of the result

[<img src="https://github.com/lipeng2/CarND-BehavioralCloning-P3/blob/master/simulation.png" height="300">](https://www.youtube.com/watch?v=NpTef1hUAn8)
