import csv
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D,Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import cv2


images = []      # a list to hold images
steering = []    # a list to hold angles

a_images = []    # a list to hold augmented images
a_steering = []  # a list to hold augmented angles

#correction = 0.2

with open('./traindata/driving_log.csv') as f:
    reader = csv.reader(f)
    for line in reader:
        steering.append(float(line[3]))
        #steering.append(float(line[3]) + correction)
        #steering.append(float(line[3]) - correction)

        # here we read the images as RGB not BGR because the drive.py uses RGB notation
        images.append(cv2.cvtColor(cv2.imread(line[0]), cv2.COLOR_BGR2RGB))
        #images.append(cv2.cvtColor(cv2.imread(line[1]), cv2.COLOR_BGR2RGB))
        #images.append(cv2.cvtColor(cv2.imread(line[2]), cv2.COLOR_BGR2RGB))


# data augmentation
for img, ster in zip(images, steering):
    a_images.append(img)
    a_images.append(cv2.flip(img, 1))
    a_steering.append(ster)
    a_steering.append(ster * -1)


# convert data to numpy arrays to work with keras
X_train = np.array(a_images)
Y_train = np.array(a_steering)

model = Sequential()

# cropping to remove useless information that confuse the model
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)))

# normalizing the data
model.add(Lambda(lambda x: x / 255.0 - 0.5))

# modified lenet5
model.add(Convolution2D(24,5,5,subsample=(2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(36,5,5,subsample=(2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(48,5,5,subsample=(2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss="mse", optimizer="adam")
model.fit(X_train, Y_train, validation_split=0.2, shuffle=True, nb_epoch=3)

model.save("model.h5")