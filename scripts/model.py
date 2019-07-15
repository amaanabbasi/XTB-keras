#!/usr/bin/env python
# coding: utf-8

# In[81]:


import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
import pickle
import json

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

#remove warnings
import warnings
import os
import tensorflow as tf
if type(tf.contrib) != type(tf): tf.contrib._warning = None
    
#ignore AVX AVX2 warning 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

warnings.filterwarnings("ignore")

# In[69]:


def finetuning_model():
    base_model=VGG16(weights='imagenet',include_top=False)

    x=base_model.output
    x=GlobalAveragePooling2D()(x)
    x=Dropout(0.25)(x)
    x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
    x=Dense(1024,activation='relu')(x) #dense layer 2
    x=Dense(512,activation='relu')(x) #dense layer 3
    preds=Dense(2,activation='softmax')(x) #final layer with softmax activation

    model=Model(inputs=base_model.input,outputs=preds)

    for layer in model.layers[:20]:
        layer.trainable=False
    for layer in model.layers[20:]:
        layer.trainable=True
    
    return model


# In[91]:


class PlotLossAcc():
    
    def __init__(self, history):
        self.history = history
        
    def plot_loss_acc(self):
        # Plot training & validation accuracy values
        plt.plot(self.history.history['acc'])
        plt.plot(self.history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        plt.savefig('history/accuracy-{}.png'.format(self.history.history['val_acc']))

        # Plot training & validation loss values
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        plt.savefig('history/loss-{}.png'.format(self.history.history['val_loss']))
   
    def save_history(self):
        json.dump(self.history.history, open('/history/history.json', 'w'))


# In[71]:


class DataGenerator():
    
    def __init__(self):
        
        self.train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies
        self.train_generator = None
        self.valid_generator = None
        self.step_size_train = None
        self.step_size_valid = None
        
    def train_test_generator(self, train_path='../data/training-data', test_path='../data/testing-data'):
        
        self.train_generator = self.train_datagen.flow_from_directory(train_path,
                                                         target_size=(64,64),
                                                         color_mode='rgb',
                                                         batch_size=32,
                                                         class_mode='categorical',
                                                         shuffle=True)
        self.valid_generator = self.train_datagen.flow_from_directory(test_path,
                                                          target_size = (64,64),
                                                          color_mode='rgb',
                                                          batch_size=12,
                                                          class_mode='categorical',
                                                          shuffle=False)
        
        self.step_size_train = self.train_generator.n//self.train_generator.batch_size
        self.step_size_valid = self.valid_generator.n//self.valid_generator.batch_size

