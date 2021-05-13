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
    width_shift_range=0.2,
    height_shift_range=0.5,
    brightness_range=(-0.2,0.2),
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    validation_split = 0.3
    )

X_train = train_datagen.flow_from_directory(
        'seg_train/seg_train',
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        subset = 'training') 

X_train.class_indices

#Test Data gen
val_datagen = ImageDataGenerator(rescale=1./255)

X_val = train_datagen.flow_from_directory(
        'seg_train/seg_train',
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        subset = 'validation') 

#Generating y_test
y_val = []
for i in X_val.filenames:
    i = i.split('\\')
    i = i[0]
    if i == "buildings":
        y_val.append('0')
        
    elif i == "forest":
        y_val.append('1')
        
    elif i == "glacier":
        y_val.append('2')
        
    elif i == "mountain":
        y_val.append('3')
        
    elif i == "sea":
        y_val.append('4')
        
    elif i == "street":
        y_val.append('5')
        
y_val = np.array(y_val)
    


#******Building the CNN*******

#Buliding a CNN model using ResNet 50 pretrained on ImageNet

from tensorflow.keras.applications.resnet50 import ResNet50
res_conv = ResNet50(include_top=False,
                    weights='imagenet',
                    input_tensor=None,
                    input_shape=(150,150,3),
                    pooling=None)#,classes=1000)



#Initializing the CNN
cnn = tf.keras.models.Sequential()

cnn.add(res_conv)

#Adding Convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters= 1024, kernel_size= 4, activation='relu', input_shape=[150,150,3]))
#cnn.add(tf.keras.layers.Conv2D(filters= 512, kernel_size= , activation='relu'))
cnn.output_shape                                  #Checking Shape
#Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))                      

# #2nd Convolutional Layer 
# cnn.add(tf.keras.layers.Conv2D(filters= 32, kernel_size= 3, activation='relu')) 
# cnn.add(tf.keras.layers.Conv2D(filters= 32, kernel_size= 3, activation='relu'))      #Convolution
# #2nd Pooling Layer
# cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2)) 

# #3rd Convo Layer
# cnn.add(tf.keras.layers.Conv2D(filters= 128, kernel_size= 3, activation='relu'))
# cnn.add(tf.keras.layers.Conv2D(filters= 128, kernel_size= 3, activation='relu'))
# #3rd Pooling  
# cnn.add(tf.keras.layers.MaxPool2D(pool_size=1, strides=1))                         #Pooling

# #4th Convo Layer
# cnn.add(tf.keras.layers.Conv2D(filters= 256, kernel_size= 3, activation='relu'))
# cnn.add(tf.keras.layers.Conv2D(filters= 256, kernel_size= 3, activation='relu'))
# #4th Pooling  
# cnn.add(tf.keras.layers.MaxPool2D(pool_size=1, strides=1))                         #Pooling


#Flattening
cnn.add(tf.keras.layers.Flatten())

cnn.output_shape 


#___Fully Connected Layer___

#Hidden Layers
cnn.add(tf.keras.layers.Dense(units=1024, activation='relu', ))

#Output Layer
cnn.add(tf.keras.layers.Dense(units=6, activation='softmax'))

#Compile 
opt = tf.keras.optimizers.RMSprop(lr=1e-4)              #
cnn.compile(optimizer= 'adam' , loss = 'categorical_crossentropy', metrics=['accuracy'])

#Training CNN on Training set and Evaluating on Test set
cnn.fit(x= X_train, validation_data= X_val, epochs= 4)


cnn.summary()

#___Saving the Model___

cnn.save('Saved_Models/model_acc_43/')
cnn.save_weights('Saved_Models/weight_acc_43/')

#Loading Model
cnn = tf.keras.models.load_model('Saved_Models/model_acc_43/')

cnn.fit(x= X_train, validation_data= X_val, epochs= 7, initial_epoch= 4 )




#_____TEST_____

#Intaking All the test images
test_datagen = ImageDataGenerator(rescale=1./255)

X_test = test_datagen.flow_from_directory(
        'seg_test/seg_test',
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical') 



#_Making prediction
y_pred_prob = cnn.predict(X_test)

#Making an array of predictions
y_pred = []

for test_image in y_pred_prob:
    
    a = np.argmax(test_image)
    
    if a == 0:
        y_pred.append('0')
        
    elif a == 1:
        y_pred.append('1')
        
    elif a == 2:
        y_pred.append('2')
        
    elif a == 3:
        y_pred.append('3')
        
    elif a == 4:
        y_pred.append('4')
        
    elif a == 5:
        y_pred.append('5')
        
y_pred = np.array(y_pred)


#Actual labels
y_act = []
for i in X_test.filenames:
    i = i.split('\\')
    i = i[0]
    if i == "buildings":
        y_act.append('0')
        
    elif i == "forest":
        y_act.append('1')
        
    elif i == "glacier":
        y_act.append('2')
        
    elif i == "mountain":
        y_act.append('3')
        
    elif i == "sea":
        y_act.append('4')
        
    elif i == "street":
        y_act.append('5')
        
y_act = np.array(y_act)

#Creating the Confusion Matrix
t_count = 0
f_count = 0
for j in range(0,len(y_act)):
    if y_pred[j]==y_act[j]:
        t_count = t_count + 1
    else:
        f_count = f_count + 1
        
print("Correctly predicted values: ",t_count,
      "Inorrectly predicted values: ",f_count)
