#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D
from keras.layers import Dropout

from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# Pretrained Models
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications import MobileNet


# In[2]:


# base_model = VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)


# In[3]:


name_base_model = "VGG16"
base_model=VGG16(weights='imagenet',include_top=False) 
#imports the mobilenet model and discards the last 1000 neuron layer.


# In[4]:


len(base_model.layers)


# In[5]:


base_model.summary()


# In[6]:


x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dropout(0.25)(x)
x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(1024,activation='relu')(x) #dense layer 2
x=Dense(512,activation='relu')(x) #dense layer 3
preds=Dense(2,activation='softmax')(x) #final layer with softmax activation


# In[7]:


model=Model(inputs=base_model.input,outputs=preds)


# In[8]:


# for i,layer in enumerate(model.layers):
#   print(i,layer.name)


# In[9]:


len(model.layers)


# In[10]:


for layer in model.layers[:20]:
    layer.trainable=False
for layer in model.layers[20:]:
    layer.trainable=True


# In[11]:


train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies

train_generator=train_datagen.flow_from_directory('data/training-data',
                                                 target_size=(64,64),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)
valid_generator = train_datagen.flow_from_directory('data/testing-data',
                                                  target_size = (64,64),
                                                  color_mode='rgb',
                                                  batch_size=12,
                                                  class_mode='categorical',
                                                  shuffle=True)


# In[13]:


# Specify parameters
lr = 0.0001
epochs = 50


# In[14]:


Adam = keras.optimizers.Adam(lr=lr)
model.compile(optimizer=Adam,loss='binary_crossentropy',metrics=['accuracy'])
# Adam optimizer
# loss function will be cbinary_crossentropy
# evaluation metric will be accuracy

step_size_train=train_generator.n//train_generator.batch_size
step_size_valid = valid_generator.n//valid_generator.batch_size
history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=step_size_train,
                              validation_data=valid_generator,
                              validation_steps=step_size_valid,
                               epochs=epochs)




# In[69]:


# Scalar test loss (if the model has a single output and no metrics) or 
#list of scalars (if the model has multiple outputs and/or metrics).
#The attribute model.metrics_names will give you the display labels for the scalar outputs.
model.evaluate_generator(generator=test_generator,steps=step_size_valid)


# In[70]:


valid_generator.reset()
pred = model.predict_generator(valid_generator, steps=step_size_test, verbose=1)
predicted_class_indices=np.argmax(pred,axis=1)
y_true = test_generator.classes




model.save("{}-{}-{}-adam.h5".format(name_base_model, epochs, lr))

