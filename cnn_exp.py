# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 11:09:39 2018

@author: sattavaram1akhil
"""

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import numpy as np
#from PIL import Image
from keras.preprocessing import image

classifier=Sequential()

classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation="relu"))

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Convolution2D(32,3,3,activation="relu"))

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Flatten())

classifier.add(Dense(output_dim=128,activation="relu"))
classifier.add(Dense(output_dim=1,activation="sigmoid"))

classifier.compile(optimizer='adam',loss = 'binary_crossentropy',metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'C:\\Users\\NIKHIL\\Downloads\\mnistasjpg\\trainingSet\\trainingSet',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

testing_set = test_datagen.flow_from_directory(
        'C:\\Users\\NIKHIL\\Downloads\\mnistasjpg\\testSet',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=100,
        epochs=20,
        validation_data=testing_set,
        validation_steps=25)


img=image.load_img("C:\\Users\\NIKHIL\\Downloads\\img_8.jpg", target_size=(64,64))
img = image.img_to_array(img)
dat=np.expand_dims(img,axis=0)
dat.shape

classifier.predict(dat)


