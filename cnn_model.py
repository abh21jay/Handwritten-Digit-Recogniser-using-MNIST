import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import Sequential 
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPool2D
from keras import optimizers
from keras.datasets import mnist
from keras.utils import to_categorical
import keras

#Loading the MNSIT DATASET 
(X_train,Y_train),(X_test,Y_test) = mnist.load_data()
print(X_train.shape,Y_train.shape,X_test.shape,Y_test.shape)

#Applying Thresholds to remove noise and clear the dataset of any confusion

_,X_train_rf = cv2.threshold(X_train,127,255,cv2.THRESH_BINARY)
_,X_test_rf = cv2.threshold(X_train,127,255,cv2.THRESH_BINARY)

#Reshaping the dataset 

X_train = X_train_rf.reshape(-1,28,28,1)
X_test = X_test_rf.reshape(-1,28,28,1)

#creating sequential inputs from 0 to 9 

Y_train = to_categorical(Y_train, num_classes = 10)
Y_test = to_categorical(Y_test, num_classes = 10)

#printing the values and crossc checking the input and output values

print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

#CNN MODEL

input_shape = (28,28,1)
number_of_classes = 10

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))
model.add(Dense(number_of_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])


model.summary()

history = model.fit(X_train, Y_train,epochs=5, shuffle=True,
                    batch_size = 200,validation_data= (X_test, Y_test))
model.save("MODEL.H5")
