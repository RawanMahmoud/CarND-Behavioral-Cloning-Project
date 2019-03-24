# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Grayscaling"
[image1]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* run2.mp4 the output video in the autonomous mode

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I usde the network which was published by NVIDIA team, this is the network they use for training a real car to drive autonomously.

the model in the code : (model.py lines 45-67) 

the model consists of five conolitional layers, each one followd by a RELU layer to introduce nonlinearity.

Then, four fully connected layers.

To decrease overfit, I added a dropout layer with 0.5 probability of discarding the weights.



#### 2. Attempts to reduce overfitting in the model

I tried to add dropout layers ro decrease the overfitting. I finally used a dropout layer after each convolutional layer with probability of discarding wegights = 0.2 and also I added a dropout layer with 0.5 probability of discarding the weights before the last fully connected layer.

The model was trained and validated on quite large data set to ensure that the model was not overfitting, I used data augmentation and collected alot of data from different positions for the car . The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 69).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to find the archeticture that gives the best accuracy.

First I was thinking of making a convolution neural network model similar to the previous project, but I found more powerful archeticture which was published by Nvidia; So I used it. I thought this model might be appropriate because they use it for training a real self driving car.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model by adding dropout layers. First the overfitting was still high.

So, I started to create new training data by recording more laps in the training mode of the simulator.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track and never return to it. to improve the driving behavior in these cases, I used more train data specially that contains recovery from left or right.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines lines 45-67) consisted of a convolution neural network with the following layers and layer sizes :

- cropping layer

- labda layer for normalizing and mean centering

- Convolution layer of size (24,5,5) followed by relu activation function followed by dropout layer with 0.2 probability of descarding weights

- Convolution layer of size (36,5,5) followed by relu activation function followed by dropout layer with 0.2 probability of descarding weights

- Convolution layer of size (48,5,5) followed by relu activation function followed by dropout layer with 0.2 probability of descarding weights

- Convolution layer of size (64,3,3) followed by relu activation function followed by dropout layer with 0.2 probability of descarding weights

- Convolution layer of size (64,3,3) followed by relu activation function followed by dropout layer with 0.2 probability of descarding weights

- Convolution layer of size (64,3,3) followed by relu activation function followed by dropout layer with 0.2 probability of descarding weights

- (FLATTEN)

- Fully connected layer with output 100

- Fully connected layer with output 50

- Fully connected layer with output 10

- Fully connected layer with output 0.5

- Fully connected layer with output 1


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to return to the center if it is not. These images show what a recovery looks like starting from when the car is at the right of the road and wants to return back to the center :

To augment the data set, I also flipped images and the angles thinking that this would help. For example, here is an image that has then been flipped:

![alt text][image1]
![alt text][image2]

This flipping helped alot in increasing the data set and generalize the model.

After the collection process, I had X number of data points . I then preprocessed this data by adding a cropping layer to remove the unwanted area (trees, sky, ...etc).
Then I added labda layer to normalize the images data and mean centering it.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced because after that the validation loss saturates or increases. I used an adam optimizer so that manually training the learning rate wasn't necessary.
