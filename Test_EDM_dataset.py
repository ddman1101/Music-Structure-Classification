#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 18:10:34 2020

@author: ddman
"""

import pandas as pd
import numpy as np
import librosa 
import os
from tqdm import tqdm
import sklearn
import random
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.metrics import classification_report
from focal_loss import SparseCategoricalFocalLoss
import matplotlib.pyplot as plt

# Collect all audio dir
all_audio_dir = os.listdir("./downloaded_audio/structure-annotation/audio/")
print(len(all_audio_dir))

# Read the label into the variable
all_labels_dir = os.listdir("./downloaded_audio/structure-annotation/annotation/")
time_labels = []
for i in range(len(all_labels_dir)):
    temp = pd.read_csv("./downloaded_audio/structure-annotation/annotation/"+all_labels_dir[i], sep=" ", header=None)
    for j in range(len(temp)):
        first = round(temp[0][j])
        second = round(temp[1][j])
        temp_ = []
        temp_.append(all_labels_dir[i])
        temp_.append(str(first)+"-"+str(second))
        temp_.append(temp[2][j])
        time_labels.append(temp_)

# Read all the audio in the list
labels_and_audio = []
for i in tqdm(range(len(time_labels))):
    y, sr = librosa.load("./downloaded_audio/structure-annotation/audio/"+time_labels[i][0][9:len(time_labels[i][0])-4]+"_"+time_labels[i][1]+".wav", sr=16000)
    # catch the time labels timestamps and crack it into 3 seconds
    mel_ = librosa.feature.melspectrogram(y, sr=sr)
    l_d = []
    l_d.append(mel_)
    l_d.append(time_labels[i][2])
    labels_and_audio.append(l_d)

# Split the segment into 3 seconds segments
a_l = []
for i in tqdm(range(len(labels_and_audio))):
    if labels_and_audio[i][0].shape[1] >= 94:
        for j in range(0, labels_and_audio[i][0].shape[1], 32):
            if labels_and_audio[i][0].shape[1] >= 94+j :
                l = []
                l.append(labels_and_audio[i][0][0:128,0+j:94+j])
                l.append(labels_and_audio[i][1])
                a_l.append(l)
                
# Split the EDM dataset into training set, testing set, validation set
proportion = [0.8, 0.1, 0.1] # Train, Test, Validation
random.shuffle(a_l)
training_p_l, testing_p_l = sklearn.model_selection.train_test_split(a_l, train_size=int(proportion[0]*len(a_l)), test_size=int(proportion[1]*len(a_l)))

# set the labels and data (training)
train_d = []
train_t = []
for i in range(len(training_p_l)):
    train_d.append(training_p_l[i][0])
    train_t.append(training_p_l[i][1])
train_d = np.array(train_d)
train_t = np.array(train_t)

# set the labels and data (testing)
test_d = []
test_t = []
for i in range(len(testing_p_l)):
    test_d.append(testing_p_l[i][0])
    test_t.append(testing_p_l[i][1])
test_d = np.array(test_d)
test_t = np.array(test_t)

# Fix the "intro" to "Intro"
for i in range(len(train_t)):
    if "intro" == train_t[i]:
        train_t[i] = "Intro"
    else :
        train_t[i] = train_t[i]
for i in range(len(test_t)):
    if "intro" == test_t[i]:
        test_t[i] = "Intro"
    else :
        test_t[i] = test_t[i]

# set the labels to the number
targets_num_train = pd.get_dummies(pd.Series(np.unique(train_t)))
targets_num_test = pd.get_dummies(pd.Series(np.unique(test_t)))

# Set the labels and data well !!!
train_targets = []
for i in range(len(train_t)):
    train_targets.append(np.argmax(np.array(targets_num_train[train_t[i]])))
train_targets = np.array(train_targets)

test_targets = []
for i in range(len(test_t)):
    test_targets.append(np.argmax(np.array(targets_num_test[test_t[i]])))
test_targets = np.array(test_targets)

val_num = len(a_l)-len(training_p_l)-len(testing_p_l)

x_val = train_d[:val_num]
y_val = train_targets[:val_num]

train_d = train_d[val_num:]
train_targets = train_targets[val_num:]

#======================================================================================================================
nb_layers = 4  # number of convolutional layers
nb_filters = [64, 128, 128, 128]  # filter sizes
kernel_size = (3, 3)  # convolution kernel size
activation = 'elu'  # activation function to use after each layer
pool_size = [(2, 2), (4, 2), (4, 2), (4, 2),
             (4, 2)]  # size of pooling area

# shape of input data (frequency, time, channels)
inputs = tf.keras.Input(shape=[128,94,1])
input_shape = (inputs.shape[1], inputs.shape[2], inputs.shape[3])
frequency_axis = 1
time_axis = 2
channel_axis = 3

# Create sequential model and normalize along frequency axis
output_1 = layers.BatchNormalization(axis=frequency_axis, input_shape=input_shape)(inputs)

# First convolution layer specifies shape
output_1 = layers.Conv2D(nb_filters[0], kernel_size=kernel_size, padding='same',
                         data_format="channels_last",
                         input_shape=input_shape)(output_1)

output_1 = layers.Activation(activation)(output_1)
output_1 = layers.BatchNormalization(axis=channel_axis)(output_1)
output_1 = layers.MaxPooling2D(pool_size=pool_size[0], strides=pool_size[0])(output_1)
output_1 = layers.Dropout(0.1)(output_1)

# Add more convolutional layers
for layer in range(nb_layers - 1):
    # Convolutional layer
    output_1 = layers.Conv2D(nb_filters[layer + 1], kernel_size=kernel_size,
                     padding='same')(output_1)
    output_1 = layers.Activation(activation)(output_1)
    output_1 = layers.BatchNormalization(
        axis=channel_axis)(output_1)  # Improves overfitting/underfitting
    output_1 = layers.MaxPooling2D(pool_size=pool_size[layer + 1],
                           strides=pool_size[layer + 1])(output_1)  # Max pooling
    output_1 = layers.Dropout(0.1)(output_1)

    # Reshaping input for recurrent layer
# (frequency, time, channels) --> (time, frequency, channel)
output_1 = layers.Permute((time_axis, frequency_axis, channel_axis))(output_1)
resize_shape = output_1.shape[2] * output_1.shape[3]
output_1 = layers.Reshape((output_1.shape[1], resize_shape))(output_1)

# recurrent layer
output_1 = layers.GRU(32, return_sequences=True)(output_1)
output_1 = layers.GRU(32, return_sequences=False)(output_1)
output_1 = layers.Dropout(0.3)(output_1)

# Output layer
output_1 = layers.Dense(50)(output_1)
output_1 = layers.Activation("softmax")(output_1)

model = tf.keras.Model(inputs, output_1)
model.summary()
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0005),
              loss=SparseCategoricalFocalLoss(gamma=2),
              metrics=['sparse_categorical_accuracy'])
#======================================================================================================================
reload_model = tf.keras.models.load_model("./model/model_focal.h5")
# es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, min_delta=0.0001)
history_transfer = reload_model.fit(train_d, train_targets, epochs=100, batch_size=16, validation_data=(x_val, y_val))

reload_model = tf.keras.models.load_model("./model/model_focal.h5")
reload_model.evaluate(test_d, test_targets)
model.evaluate(test_d, test_targets)

reload_model.save('./model/model_transfer.h5')

# test the confusion matrix
test_predict = reload_model.predict(test_d, batch_size=16)
test_pred_bool = np.argmax(test_predict, axis=1)
columns_n = pd.DataFrame(test_t).value_counts()
report = pd.DataFrame(classification_report(test_targets, test_pred_bool, output_dict = True)).T
temp_support = report["support"]

columns_name = []
for i in range(len(temp_support)):
    for j in range(len(columns_n)):
        if int(columns_n[columns_n.index[j][0]][0]) == int(temp_support[i]):
            print(columns_n.index[j][0])
            columns_name.append(columns_n.index[j][0])
        else:
            columns_name = columns_name
            
print(classification_report(test_targets, test_pred_bool, target_names = columns_name))
reports = pd.DataFrame(classification_report(test_targets, test_pred_bool, output_dict = True, target_names = columns_name)).T
reports
reports.to_csv("reports_transfer.csv", sep="\t")

confusion_matrix = pd.DataFrame(sklearn.metrics.confusion_matrix(test_targets, test_pred_bool))
#======================================================================================================================

# summarize history for accuracy
plt.plot(history_transfer.history['loss'])
plt.plot(history_transfer.history['val_loss'])
plt.title('Training loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss', 'val loss'], loc='lower left')
plt.show()
# summarize history for loss plt.plot(history.history['loss']) plt.plot(history.history['val_loss']) plt.title('model loss')
plt.plot(history_transfer.history['sparse_categorical_accuracy'])
plt.plot(history_transfer.history['val_sparse_categorical_accuracy'])
plt.title('Training accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train accuracy', 'val accuracy'], loc='upper left')
plt.show()