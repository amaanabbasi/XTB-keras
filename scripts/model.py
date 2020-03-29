#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
import pickle
import json

from keras.layers import Dense,GlobalAveragePooling2D, BatchNormalization
from keras.layers import Dropout, Activation
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# Pretrained Models
from keras.applications import VGG19
from keras.applications import VGG16
from keras.applications import ResNet50
from keras.applications import MobileNet


# In[63]:


def finetuning_model(batch_normalization=True):
    # IN_SHAPE = (224, 224, 3)
    base_model=VGG16(include_top=False, weights="imagenet") #  input_shape=IN_SHAPE
    x=base_model.output
    
    if batch_normalization:
        x=GlobalAveragePooling2D()(x)
        x=Dropout(0.25)(x)

        x=Dense(1024)(x) 
        x=BatchNormalization()(x)
        x=Activation("relu")(x)

        x=Dense(1024)(x) 
        x=BatchNormalization()(x)
        x=Activation("relu")(x)

        x=Dense(512)(x) 
        x=BatchNormalization()(x)
        x=Activation("relu")(x)
        
    else:
        x=GlobalAveragePooling2D()(x)
        x=Dropout(0.25)(x)
        x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
        x=Dense(1024,activation='relu')(x) #dense layer 2
        x=Dense(512,activation='relu')(x) #dense layer 3
    
    preds=Dense(2,activation='softmax')(x) #final layer with softmax activation

    model=Model(inputs=base_model.input,outputs=preds)

    for layer in model.layers[:19]:
        layer.trainable=False
    for layer in model.layers[19:]:
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
        plt.savefig('accuracy-{}.png'.format(self.history.history['val_acc']))
        try:
            plt.show()
        except Exception as err:
            print(err)
        # Plot training & validation loss values
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig('loss-{}.png'.format(self.history.history['val_loss']))
        try:
            plt.show()
        except Exception as err:
            print(err)
    
    def save_history(self):
        json.dump(self.history.history, open('../history/history.json', 'w'))


# In[71]:


class DataGenerator():
    
    def __init__(self):
        
        self.datagenerator = ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies
        self.train_generator = None
        self.valid_generator = None
        self.step_size_train = None
        self.step_size_valid = None
        
    def train_valid_generator(self, train_path='../data/training-data', test_path='../data/testing-data'):
        
        self.train_generator = self.datagenerator.flow_from_directory(train_path,
                                                         target_size=(64,64),
                                                         color_mode='rgb',
                                                         batch_size=32,
                                                         class_mode='categorical',
                                                         shuffle=True)
        self.valid_generator = self.datagenerator.flow_from_directory(test_path,
                                                          target_size = (64,64),
                                                          color_mode='rgb',
                                                          batch_size=12,
                                                          class_mode='categorical',
                                                          shuffle=False)
        
        self.step_size_train = self.train_generator.n//self.train_generator.batch_size
        self.step_size_valid = self.valid_generator.n//self.valid_generator.batch_size
    
    def test_generator(self, test_path='../data/testing-data'):       
        
        self.test_generator = self.datagenerator.flow_from_directory(test_path,
                                                         target_size=(64,64),
                                                         color_mode='rgb',
                                                         batch_size=32,
                                                         class_mode='categorical',
                                                         shuffle=True)
        
        


# In[105]:


# def print_stats(model, epochs, lr):
#     print("epochs: {}, learning rate: {}".format(epochs, lr))
#     print()
#     print(model.summary())


# # In[ ]:


# d = DataGenerator()
# d.train_test_generator()

# epochs = 1
# lr = 1e-4

# Adam = keras.optimizers.Adam(lr=lr)

# model = finetuning_model()
# print_stats(model, epochs, lr)
# model.compile(optimizer=Adam,loss='binary_crossentropy',metrics=['accuracy'])

# history = model.fit_generator(generator=d.train_generator,
#                               steps_per_epoch=d.step_size_train,
#                               validation_data=d.valid_generator,
#                               validation_steps=d.step_size_valid,
#                               epochs=epochs)

# model.save('VGG16-{}-{}-adam.h5'.format(epochs, lr,))
# h = PlotLossAcc(history)
# h.plot_loss_acc()
# h.save_history()


# In[64]:


# model = finetuning_model()


# # In[66]:


# model.summary()


# # In[72]:


# for layer in model.layers:
#     print(f"""layer-name={layer.name}, trainable={layer.trainable}""")        

