import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import math as mt

#Train Datagen
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
   #brightness_range=(-0.2,0.2),
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split = 0.3
    )

X_train = train_datagen.flow_from_directory(
        'train',
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        subset = 'training') 

X_train.class_indices



#Test Data gen
val_datagen = ImageDataGenerator(rescale=1./255)

X_val = train_datagen.flow_from_directory(
        'train',
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        subset = 'validation') 

#******Building the CNN*******

#Initializing the CNN
cnn = tf.keras.models.Sequential()

#Convolution
cnn.add(tf.keras.layers.Conv2D(filters= 128, kernel_size= 3, activation='relu', input_shape=[150, 150, 3]))

#Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))                       

#2nd Convolutional Layer 
cnn.add(tf.keras.layers.Conv2D(filters= 64, kernel_size= 3, activation='relu'))       #Convolution
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))                            #Pooling


#Flattening
cnn.add(tf.keras.layers.Flatten())
cnn.output_shape


#Fully Connected Layer
#Hidden Layers
cnn.add(tf.keras.layers.Dense(units=1024, activation='relu', kernel_initializer='glorot_uniform'))
cnn.add(tf.keras.layers.Dense(units=512, activation='relu'))

#Output Layer
cnn.add(tf.keras.layers.Dense(units=3, activation='softmax'))

#Compile
cnn.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

#Training CNN on Training set and Evaluating on Test set
cnn.fit(x= X_train, validation_data= X_val, epochs= 15)

#SAVING

cnn.save('MODELS/model_acc_/')
