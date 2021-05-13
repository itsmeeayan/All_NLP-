import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#tf.__version__

train_datagen = ImageDataGenerator(
        rescale=1./255,
        #Image Augmentation
        shear_range=0.2,
        rotation_range = 40,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True)

train_set = train_datagen.flow_from_directory(
        'plantvillage',
        target_size=(256, 256),
        batch_size=32,
        class_mode='categorical') 

#Test Data gen
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
        'Val',
        target_size=(256, 256),
        batch_size=32,
        class_mode='categorical')

#******Building the CNN*******

#Initializing the CNN
cnn = tf.keras.models.Sequential()

#Convolution
cnn.add(tf.keras.layers.Conv2D(filters= 32, kernel_size= 3, activation='relu', input_shape=[256,256,3]))

#Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))                       

#2nd Convolutional Layer 
cnn.add(tf.keras.layers.Conv2D(filters= 32, kernel_size= 3, activation='relu'))       #Convolution
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))                            #Pooling


#Flattening
cnn.add(tf.keras.layers.Flatten())

#Fully Connected Layer

#Hidden Layers
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

#Output Layer
cnn.add(tf.keras.layers.Dense(units=10, activation='softmax'))

#Compile
cnn.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

#Training CNN on Training set and Evaluating on Test set
cnn.fit(x= train_set, validation_data= test_set, epochs= 6)

#SINGLE OUTPUT

from tensorflow.keras.preprocessing import image
sample_image = image.load_img('sample/Septoria_leaf_spot (189).jpg', target_size=(256,256))
#plt.imshow(sample_image)
sample_image = image.img_to_array(sample_image)
sample_image = np.expand_dims(sample_image, axis = 0 )
result = cnn.predict(sample_image/255.0)

#Revealing the indices for each class
train_set.class_indices

print(result)

#Displaying the reasult
a = ""
a = np.argmax(result)
print(a)

if a == 1:
    pred = "Early Blight"
elif a == 0:
    pred = "Bacterial spot"
elif a == 2:
    pred = "Late_blight"
elif a == 3:
    pred = "Leaf_Mold"
elif a == 4:
    pred = "Septoria_leaf_spot"
elif a == 5:
    pred = "Spider_mites Two-spotted_spider_mite"
elif a == 6:
    pred = "Target_Spot"
elif a == 7:
    pred = "Tomato_Yellow_Leaf_Curl_Virus"
elif a == 8:
    pred = "Tomato_mosaic_virus"
elif a == 9:
    pred = "healthy"

print(pred)
