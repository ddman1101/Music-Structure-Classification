#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 13:58:51 2020

@author: ddman
"""

import pandas as pd
import librosa
import numpy as np
import os
import random
import sklearn
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.metrics import classification_report
from focal_loss import SparseCategoricalFocalLoss
from tqdm import tqdm
import matplotlib.pyplot as plt

"""

This area is about reading data and merge the data into a variable: final_patch_and_labels for calculating

"""

# music_data = pd.read_csv("salami_youtube_pairings.csv", sep=',')
# all_ids = music_data[['salami_id','youtube_id']]

all_music_data = pd.read_csv("../salami-data-public/metadata/SALAMI_iTunes_library.csv", sep=',')
all_music_data_2 = pd.read_csv("./salami_youtube_pairings.csv", sep=",")

# Calculate read the mode of Bit rate
# bitrate = int(all_music_data['Bit Rate'].mode())
music_id = all_music_data['salami_id']

# intersection the ids (all music data ids is different from all music data 2)
music_ids = set(all_music_data['salami_id']).intersection(set(all_music_data_2['salami_id']))

# find all music with ids
music_list = []
all_music = os.listdir("./downloaded_audio/")
for i in range(len(all_music)):
    if ".mp3" == os.path.splitext(all_music[i])[-1]:
        music_list.append(all_music[i])
    else :
        music_list = music_list
        
# # finds all ids     
# all_music_ids = []        
# for j in range(len(music_list)):
#     all_music_ids.append(music_list[j].split('-')[-1].split('.')[0])

# combine the number and mp3 file name
music_path = []
for i in range(len(all_ids["salami_id"])):
    for j in range(len(music_list)):
        if os.path.splitext(music_list[j])[-2][-11:] == all_ids["youtube_id"][i]:
            temp_list = []
            temp_list.append(all_ids["salami_id"][i])
            temp_list.append(music_list[j])
            music_path.append(temp_list)
        else:
            music_path = music_path

# interaction the music filepath and the salami id
music_all_path = []
for i in range(len(music_ids)):
    for j in range(len(music_path)):
        if music_path[j][0] == list(music_ids)[i]:
            music_all_path.append(music_path[j])
        else:
            music_all_path = music_all_path

# crab all the path of labels into the list
temp_all = os.listdir("../salami-data-public/annotations/")
for i in range(len(temp_all)):
    temp_all[i] = int(temp_all[i])
temp_all.sort() # number in the annotation
for i in range(len(music_all_path)):
    for j in range(len(temp_all)):
        if temp_all[j] == music_all_path[i][0]:
            if os.path.isfile("../salami-data-public/annotations/{}/parsed/textfile1_functions.txt".format(music_all_path[i][0])):
                music_all_path[i].append(pd.read_csv("../salami-data-public/annotations/{}/parsed/textfile1_functions.txt".format(music_all_path[i][0]), sep="\t",  header=None))
            else:
                music_all_path[i].append(pd.read_csv("../salami-data-public/annotations/{}/parsed/textfile2_functions.txt".format(music_all_path[i][0]), sep="\t",  header=None))
        else:
            music_all_path = music_all_path
            
final_all_path_and_labels = []    
for i in range(len(music_all_path)):
    if len(music_all_path[i]) != 3 :
        final_all_path_and_labels = final_all_path_and_labels
    else :
        final_all_path_and_labels.append(music_all_path[i])

# calculate the genre in the final_all_path_and_labels
for i in range(len(all_music_data)):
    for j in range(len(final_all_path_and_labels)):
        # find out the music name 
        if final_all_path_and_labels[j][0] == all_music_data["salami_id"][i]:
            final_all_path_and_labels[j].append(all_music_data["Genre"][i])
        else :
            final_all_path_and_labels = final_all_path_and_labels

# calculate the number categories of labels
all_labels = []
for i in range(len(final_all_path_and_labels)):
    all_labels.append(final_all_path_and_labels[i][2])

num_labels = []
for i in range(len(all_labels)):
    num_labels.append(all_labels[i][1].value_counts())

all_labels_and_freq = pd.concat(num_labels, axis=1)
all_labels_and_freq = all_labels_and_freq.fillna(0)
all_labels_and_freq.sum(axis=1)

all_labels_and_freq.to_csv("original_num_of_categories.csv", sep="\t") # Save the number of categories of labels data
all_labels_and_freq.sum(axis=1).to_csv("original_sum_of_categories.csv", sep="\t") # Save the sum of categories of labels data

#=======================================================================================================================================
# rest_labels = ['Verse', 'Silence', 'Chorus', 'End', 'Outro', 'Intro', 'Bridge', 'Intrumental', 'Interlude', 'Fade-out',\
#                'Solo', 'Pre-Verse', 'silence', 'Pre-Chorus', 'Head', 'Coda', 'Theme', 'Transition',\
#                'Main_Theme', 'Development', 'Secondary_theme', 'Secondary_Theme', 'outro']
# # read the audio and transform to Mel-spectrograms and cut it into patches 
# patch_with_labels_list = []
# for i in range(len(final_all_path_and_labels)):
#     y, sr = librosa.load("./downloaded_audio/{}".format(final_all_path_and_labels[i][1]),sr=16000)
#     # mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
#     # Get the sample point in timestamps
#     s_array = librosa.time_to_samples(np.array(final_all_path_and_labels[i][2][0]), sr=16000)
#     # Check the labels and deal with it!!! 
#     for j in range(len(final_all_path_and_labels[i][2][1])):
#         if final_all_path_and_labels[i][2][1][j] == "no_function":
#             temp_ = 1
#             while temp_ < 10 :
#                 if final_all_path_and_labels[i][2][1][j-1] != "no_function":
#                     final_all_path_and_labels[i][2][1][j] = final_all_path_and_labels[i][2][1][j-1]
#                     temp_ += 10
#                 else : 
#                     j -= 1
#         else :
#             final_all_path_and_labels[i][2][1][j] = final_all_path_and_labels[i][2][1][j]
#     # Use label with the audio(y)
#     for k in range(len(final_all_path_and_labels[i][2][1])-1):
#         l_list = []
#         l_list.append(final_all_path_and_labels[i][2][1][k])
#         if s_array[k] != s_array[k+1] and s_array[k+1] <= np.shape(y)[0]:
#             mel_spec = librosa.feature.melspectrogram(y=y[s_array[k]:s_array[k+1]], sr=sr)
#             l_list.append(mel_spec)
#             patch_with_labels_list.append(l_list)
#         else :
#             patch_with_labels_list = patch_with_labels_list
#     print(i)

#===========================================================================================================================================================

# Then we delete the short segment and deal with all same labels (ex: outro and Outro),
# and calculate the all_labels_and_freq again.

# step 1 : crack the data to the segments 
seg = []
for i in range(len(final_all_path_and_labels)):
    path = final_all_path_and_labels[i][1]
    seg_ = final_all_path_and_labels[i][2]
    genre = final_all_path_and_labels[i][3]
    # segment the msuic timestamps data
    for j in range(len(seg_)-2):
        temp_seg = []
        temp_seg.append(path)
        temp_seg.append(np.array(seg_[0][j+0:j+2]))
        temp_seg.append(seg_[1][j])
        temp_seg.append(genre)
        seg.append(temp_seg)
        
# step 2 : eliminate some labels (ex: no_function, count-in, vocal etc.) and the short parts
rest_labels = ['Verse', 'Silence', 'Chorus', 'End', 'Outro', 'Intro', 'Bridge', 'Intrumental', 'Interlude', 'Fade-out',\
                'Solo', 'Pre-Verse', 'silence', 'Pre-Chorus', 'Head', 'Coda', 'Theme', 'Transition',\
                'Main_Theme', 'Development', 'Secondary_theme', 'Secondary_Theme', 'outro']
all_seg = []
for i in range(len(seg)):
    for label in rest_labels:
        if seg[i][2] == label :
            if (seg[i][1][1]-seg[i][1][0]) > 3 : # short segments are eliminated
                all_seg.append(seg[i])
            else :
                all_seg = all_seg
        else :
            all_seg = all_seg

# step 3 : add the same labels together (ex: outro and Outro, Secondary_theme and Secondary_Theme) 
for i in range(len(all_seg)):
    if all_seg[i][2] == 'silence' :
        all_seg[i][2] = 'Silence'
    elif all_seg[i][2] == 'outro' :
        all_seg[i][2] = 'Outro'
    elif all_seg[i][2] == 'Secondary_theme' :
        all_seg[i][2] = 'Secondary_Theme'
    else :
        all_seg[i][2] = all_seg[i][2]
        
print(len(all_seg))


for i in tqdm(range(len(all_seg))):
    if len(all_seg[i]) == 4 :
        y, sr = librosa.load("./downloaded_audio/{}".format(all_seg[i][0]),sr=16000)
        t_array = []
        for j in range(len(all_seg)):
            if all_seg[i][0] == all_seg[j][0]:
                t_array.append(j)
            else :
                t_array = t_array
        for k in range(len(t_array)):
            s_array = librosa.time_to_samples(np.array(all_seg[t_array[k]][1]), sr=16000)
            if len(y) >= s_array[1]:
                mel_spec = librosa.feature.melspectrogram(y=y[s_array[0]:s_array[1]], sr=sr)
                all_seg[t_array[k]].append(mel_spec)
            else :
                all_seg = all_seg
    else :
        all_seg = all_seg
    print("progress:{}%".format(round((i/len(all_seg))*100, 3)))
    
# fix the wrong segment (ex: len(all_seg)!=4)
new_seg = []
for i in range(len(all_seg)):
    if len(all_seg[i]) != 4:
        new_seg = new_seg
    else:
        new_seg.append(all_seg[i])
# calculate thhe genre (with patch (not 3 seconds) )
allseg = pd.DataFrame(all_seg, columns=["name", "timestamps", "labels", "genre"])
allseg["genre"].value_counts().to_csv("labels_with_genre-(patch_with_segments).csv", sep="\t")
for i in allseg["labels"].value_counts().index:
    allseg["genre"][allseg["labels"] == "{}".format(i)].value_counts().to_csv("./genre-patch-statistics/{}-genre-distribution.csv".format(i), sep="\t")
#=======================================================================================================================================

small_labels = ["Intro", "Verse", "Chorus", "Bridge", "Theme", "Solo", "Outro"]

small_seg = []
for i in range(len(all_seg)):
    for label in small_labels:
        if all_seg[i][2] == label :
            if (all_seg[i][1][1]-all_seg[i][1][0]) > 3 : # short segments are eliminated
                small_seg.append(all_seg[i])
            else :
                small_seg = small_seg
        else :
            small_seg = small_seg

# step 3 : add the same labels together (ex: outro and Outro, Secondary_theme and Secondary_Theme) 
# for i in range(len(all_seg)):
#     if all_seg[i][2] == 'silence' :
#         all_seg[i][2] = 'Silence'
#     elif all_seg[i][2] == 'outro' :
#         all_seg[i][2] = 'Outro'
#     elif all_seg[i][2] == 'Secondary_theme' :
#         all_seg[i][2] = 'Secondary_Theme'
#     else :
#         all_seg[i][2] = all_seg[i][2]
        
# print(len(all_seg))


for i in tqdm(range(len(small_seg))):
    if len(small_seg[i]) == 4 :
        y, sr = librosa.load("./downloaded_audio/{}".format(small_seg[i][0]),sr=16000)
        t_array = []
        for j in range(len(small_seg)):
            if small_seg[i][0] == small_seg[j][0]:
                t_array.append(j)
            else :
                t_array = t_array
        for k in range(len(t_array)):
            s_array = librosa.time_to_samples(np.array(small_seg[t_array[k]][1]), sr=16000)
            if len(y) >= s_array[1]:
                mel_spec = librosa.feature.melspectrogram(y=y[s_array[0]:s_array[1]], sr=sr)
                small_seg[t_array[k]].append(mel_spec)
            else :
                small_seg = small_seg
    else :
        small_seg = small_seg
    
#=======================================================================================================================================
# fix the wrong segment (ex: len(all_seg)!=4)
new_seg = []
for i in range(len(small_seg)):
    if len(small_seg[i]) != 5:
        new_seg = new_seg
    else:
        new_seg.append(small_seg[i])

# calculate thhe genre (with patch (not 3 seconds) )
allseg = pd.DataFrame(small_seg, columns=["name", "timestamps", "labels", "genre"])
allseg["genre"].value_counts().to_csv("labels_with_genre-(patch_with_segments).csv", sep="\t")
for i in allseg["labels"].value_counts().index:
    allseg["genre"][allseg["labels"] == "{}".format(i)].value_counts().to_csv("./genre-patch-statistics/{}-genre-distribution.csv".format(i), sep="\t")

#=========================================================================================================================================

g_l = []
for i in range(len(all_seg)):
    for j in range(int(all_seg[i][1][0]), int(all_seg[i][1][1])-2, 1):
        segment = []
        segment.append(all_seg[i][2])
        segment.append(all_seg[i][3])
        g_l.append(segment)

g_l_statis = pd.DataFrame(g_l, columns=["labels", "genre"])

for i in g_l_statis["labels"].value_counts().index:
    g_l_statis["genre"][g_l_statis["labels"] == "{}".format(i)].value_counts().to_csv("./genre-patch-statistics/{}-genre-distribution(3-seconds).csv".format(i), sep="\t")
#=========================================================================================================================================
# fill all the long enough patches into the new list and crack it into 3 seconds patches
p_l = []
for i in range(len(new_seg)):
    if np.shape(new_seg[i][4])[1] >= 94 :
        # crack it into 3 seconds patches
        for j in range(0, np.shape(new_seg[i][4])[1], 32):
            l = []
            if np.shape(new_seg[i][4])[1] >= 94+j :
                temp_array = new_seg[i][3][0:128,0+j:94+j]
                l.append(new_seg[i][2])
                l.append(temp_array)
                p_l.append(l)
            else :
                p_l = p_l
    else:
        p_l = p_l

# # Read the p_l into python
np.savez("p_l.npz", np.array(p_l))
a_l = np.load("p_l.npz", allow_pickle="True")
t_l  = a_l['arr_0'] 
p_l = t_l
# # calculate all data labels porportion (p_l) ----->(labels_proportion)
all_p_l = pd.DataFrame(t_l, columns = ["labels", "Data"])
all_p_l = list(all_p_l)
# all_p_l.to_csv("all_patched_and_labels.csv", sep="\t")
# all_p_l["labels"].value_counts().to_csv("all_patches_and_labels_distribution.csv", sep = "\t")
labels_sum = all_p_l["labels"].value_counts()
labels_proportion = all_p_l["labels"].value_counts()/np.sum(all_p_l["labels"].value_counts())

# aa = 0
# for i in range(len(p_l)):
#     aa += len(p_l[i][1])
# print(aa) # Total the patches number
# to_l = []
# for i in range(len(p_l)):
#     to_l.append(p_l[i][0])
# len(np.unique(to_l)) # Total the patches labels number
# targets_num = pd.get_dummies(pd.Series(np.unique(to_l))) # Get the labels to the number(arrays)

#===========================================================================================================================================
# shuffle and split the data set to test set and train set (Care about the proportion)
train_test_val = [0.8, 0.1, 0.1]
training_p_l = []
testing_p_l = []
validation_p_l = []
all_the_position = []

for i in labels_proportion.index: # find all labels position
    temp_position = []
    l = []
    l.append(i)
    for j in range(len(p_l)):
        if p_l[j][0] == i :
            temp_position.append(j)
        else :
            temp_position = temp_position
    l.append(temp_position)
    all_the_position.append(l)

# distribute the labels to "train", "test", "validation"
final_number_of_labels = []
for i in range(len(all_the_position)): # set the "training_p_l" first the cut the rest into two parts
    for j in range(int(len(all_the_position[i][1])*train_test_val[0])):
        training_p_l.append(p_l[all_the_position[i][1][j]])
    final_number_of_labels.append(j)

final_num = []
for i in range(len(all_the_position)): # set the "testing_p_l" and "validation_p_l"
    for k in range(int(len(all_the_position[i][1])*train_test_val[1])):
        testing_p_l.append(p_l[all_the_position[i][1][final_number_of_labels[i]+k]])
    final_num.append(k+final_number_of_labels[i])
        
for i in range(len(all_the_position)): # set the "testing_p_l" and "validation_p_l"
    for p in range(int(len(all_the_position[i][1])*train_test_val[2])):
        validation_p_l.append(p_l[all_the_position[i][1][final_num[i]+p]])
        
    

# shuffle and split the data set to test set and train set (Don't care about the porportion)
random.shuffle(p_l)
training_p_l, testing_p_l = sklearn.model_selection.train_test_split(p_l, train_size=len(p_l)-10000, test_size=10000)

# set the labels and data (training)
train_d = []
train_t = []
for i in range(len(training_p_l)):
    train_d.append(training_p_l[i][1])
    train_t.append(training_p_l[i][0])
train_d = np.array(train_d)
train_t = np.array(train_t)

# set the labels and data (testing)
test_d = []
test_t = []
for i in range(len(testing_p_l)):
    test_d.append(testing_p_l[i][1])
    test_t.append(testing_p_l[i][0])
test_d = np.array(test_d)
test_t = np.array(test_t)

# set the labels and data (testing)
val_d = []
val_t = []
for i in range(len(validation_p_l)):
    val_d.append(validation_p_l[i][1])
    val_t.append(validation_p_l[i][0])
val_d = np.array(val_d)
val_t = np.array(val_t)

# testing_dataset = tf.data.Dataset.from_tensor_slices((test_d, test_t))

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

val_targets = []
for i in range(len(val_t)):
    val_targets.append(np.argmax(np.array(targets_num_test[val_t[i]])))
val_targets = np.array(val_targets)

x_val = train_d[:10000]
y_val = train_targets[:10000]

train_d = train_d[10000:]
train_targets = train_targets[10000:]

# training_dataset = tf.data.Dataset.from_tensor_slices((train_d, train_targets))
# testing_dataset = tf.data.Dataset.from_tensor_slices((test_d, test_targets))

# BATCH_SIZE = 10
# SHUFFLE_BUFFER_SIZE = 100

# training_dataset = training_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
# testing_dataset = testing_dataset.batch(BATCH_SIZE)

#=======================================================================
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

# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, min_delta=0.0001)
history = model.fit(train_d, train_targets, epochs=2000, batch_size=16, validation_data=(val_d, val_targets), callbacks=[es])
model.save_weights("./model/model_focal.weight")
model.save('./model/model_focal.h5')

# Test the accuracy
model.evaluate(test_d, test_targets)

# calculate the other indicator (F1, AUC, confusion matrix, Recall, Precision)
test_predict = model.predict(test_d, batch_size=16)
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
reports.to_csv("reports_4.csv", sep="\t")

# calculate the other indicator (F1, AUC, confusion matrix, Recall, Precision)

model.save_weights("./model/model_.weight")
model.save('./model/model_2.h5')

# reload the model and keep to train it
reload_model = tf.keras.models.load_model("./model/model.h5")
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, min_delta=0.0001)
history_3 = reload_model.fit(train_d, train_targets, epochs=2000, batch_size=16, validation_data=(x_val, y_val), callbacks=[es])
reload_model.evaluate(test_d, test_targets)

reload_model.save('./model/model_2.h5')
reload_model.save_weights("./model/model_2.weight")

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
reports.to_csv("reports_2.csv", sep="\t")

confusion_matrix = pd.DataFrame(sklearn.metrics.confusion_matrix(test_targets, test_pred_bool))

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss', 'val loss'], loc='lower left')
plt.show()
# summarize history for loss plt.plot(history.history['loss']) plt.plot(history.history['val_loss']) plt.title('model loss')
plt.plot(history.history['sparse_categorical_accuracy'])
plt.plot(history.history['val_sparse_categorical_accuracy'])
plt.title('Training accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train accuracy', 'val accuracy'], loc='upper left')
plt.show()

print(history_set.history_set.keys())
# summarize history for accuracy
plt.plot(history_set.history_set['loss'])
plt.plot(history_set.history_set['val_loss'])
plt.title('Training loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss', 'val loss'], loc='lower left')
plt.show()
# summarize history for loss plt.plot(history.history['loss']) plt.plot(history.history['val_loss']) plt.title('model loss')
plt.plot(history.history['sparse_categorical_accuracy'])
plt.plot(history.history['val_sparse_categorical_accuracy'])
plt.title('Training accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train accuracy', 'val accuracy'], loc='upper left')
plt.show()



plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#=======================================================================
# i = 2
# inputs = tf.keras.Input(shape=[128,94,1])

# temp1 = layers.Conv2D(i*10, (1,100), padding='same', activation='relu')(inputs)
# temp1 = layers.MaxPooling2D((4,temp1.shape[2]))(temp1)

# temp2 = layers.Conv2D(i*6, (3,100), padding='same', activation='relu')(inputs)
# temp2 = layers.MaxPooling2D((4,temp2.shape[2]))(temp2)

# temp3 = layers.Conv2D(i*3, (5,100), padding='same', activation='relu')(inputs)
# temp3 = layers.MaxPooling2D((4,temp3.shape[2]))(temp3)

# temp4 = layers.Conv2D(i*3, (7,100), padding='same', activation='relu')(inputs)
# temp4 = layers.MaxPooling2D((4,temp4.shape[2]))(temp4)

# temp5 = layers.Conv2D(i*15, (1,75), padding='same', activation='relu')(inputs)
# temp5 = layers.MaxPooling2D((4,temp5.shape[2]))(temp5)

# temp6 = layers.Conv2D(i*10, (3,75), padding='same', activation='relu')(inputs)
# temp6 = layers.MaxPooling2D((4,temp6.shape[2]))(temp6)

# temp7 = layers.Conv2D(i*5, (5,75), padding='same', activation='relu')(inputs)
# temp7 = layers.MaxPooling2D((4,temp7.shape[2]))(temp7)

# temp8 = layers.Conv2D(i*5, (7,75), padding='same', activation='relu')(inputs)
# temp8 = layers.MaxPooling2D((4,temp8.shape[2]))(temp8)

# temp9 = layers.Conv2D(i*15, (1,25), padding='same', activation='relu')(inputs)
# temp9 = layers.MaxPooling2D((4,temp9.shape[2]))(temp9)

# temp10 = layers.Conv2D(i*10, (3,25), padding='same', activation='relu')(inputs)
# temp10 = layers.MaxPooling2D((4,temp10.shape[2]))(temp10)

# temp11 = layers.Conv2D(i*5, (5,25), padding='same', activation='relu')(inputs)
# temp11 = layers.MaxPooling2D((4,temp11.shape[2]))(temp11)

# temp12 = layers.Conv2D(i*5, (7,25), padding='same', activation='relu')(inputs)
# temp12 = layers.MaxPooling2D((4,temp12.shape[2]))(temp12)

# temp = layers.Concatenate()([temp1,temp2,temp3,temp4,temp5,temp6,temp7,temp7,temp8,temp9,temp10,temp11,temp12])
# temp = layers.Reshape((temp.shape[3],temp.shape[1],temp.shape[2]))(temp)
# temp = layers.Conv2D(i*16, (8,1), activation='relu')(temp)
# temp = layers.MaxPooling2D((4,1))(temp)
# temp = layers.Flatten()(temp)
# temp = layers.AlphaDropout(0.5)(temp)
# temp = layers.Dense(100, activation='relu')(temp)
# temp = layers.AlphaDropout(0.5)(temp)
# outputs = layers.Dense(39, activation='softmax')(temp)

# model = tf.keras.Model(inputs, outputs)
# model.summary()

# model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0005),
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['sparse_categorical_accuracy'])

# model.fit(training_dataset, epochs=300)

# model.evaluate(testing_dataset)

# #=====================================================================================================================
# i = 2
# inputs = tf.keras.Input(shape=[128,94,1])
# frequency_axis = 1
# time_axis = 2
# channel_axis = 3

# temp1 = layers.Conv2D(i*10, (1,100), padding='same', activation='relu')(inputs)
# temp1 = layers.MaxPooling2D((4,temp1.shape[2]))(temp1)

# temp2 = layers.Conv2D(i*6, (3,100), padding='same', activation='relu')(inputs)
# temp2 = layers.MaxPooling2D((4,temp2.shape[2]))(temp2)

# temp3 = layers.Conv2D(i*3, (5,100), padding='same', activation='relu')(inputs)
# temp3 = layers.MaxPooling2D((4,temp3.shape[2]))(temp3)

# temp4 = layers.Conv2D(i*3, (7,100), padding='same', activation='relu')(inputs)
# temp4 = layers.MaxPooling2D((4,temp4.shape[2]))(temp4)

# temp5 = layers.Conv2D(i*15, (1,75), padding='same', activation='relu')(inputs)
# temp5 = layers.MaxPooling2D((4,temp5.shape[2]))(temp5)

# temp6 = layers.Conv2D(i*10, (3,75), padding='same', activation='relu')(inputs)
# temp6 = layers.MaxPooling2D((4,temp6.shape[2]))(temp6)

# temp7 = layers.Conv2D(i*5, (5,75), padding='same', activation='relu')(inputs)
# temp7 = layers.MaxPooling2D((4,temp7.shape[2]))(temp7)

# temp8 = layers.Conv2D(i*5, (7,75), padding='same', activation='relu')(inputs)
# temp8 = layers.MaxPooling2D((4,temp8.shape[2]))(temp8)

# temp9 = layers.Conv2D(i*15, (1,25), padding='same', activation='relu')(inputs)
# temp9 = layers.MaxPooling2D((4,temp9.shape[2]))(temp9)

# temp10 = layers.Conv2D(i*10, (3,25), padding='same', activation='relu')(inputs)
# temp10 = layers.MaxPooling2D((4,temp10.shape[2]))(temp10)

# temp11 = layers.Conv2D(i*5, (5,25), padding='same', activation='relu')(inputs)
# temp11 = layers.MaxPooling2D((4,temp11.shape[2]))(temp11)

# temp12 = layers.Conv2D(i*5, (7,25), padding='same', activation='relu')(inputs)
# temp12 = layers.MaxPooling2D((4,temp12.shape[2]))(temp12)

# temp = layers.Concatenate()([temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp7, temp8, temp9, temp10, temp11, temp12])

# resize_shape = temp.shape[2] * temp.shape[3]
# # temp = layers.Reshape((temp.shape[1], resize_shape))(temp)
# temp = layers.Reshape((temp.shape[3],temp.shape[1],temp.shape[2]))(temp)
# temp = layers.Conv2D(i*16, (8,1), activation='relu')(temp)
# output_1 = layers.MaxPooling2D((4,1))(temp)
# output_1 = layers.Reshape((output_1.shape[1],output_1.shape[2]*output_1.shape[3]))(output_1)

# # recurrent layer
# temp = layers.GRU(128, return_sequences=True)(output_1)
# temp = layers.GRU(128, return_sequences=False)(output_1)
# temp = layers.Dropout(0.3)(temp)

# outputs = layers.Dense(39, activation='softmax')(temp)

# model = tf.keras.Model(inputs, outputs)
# model.summary()
# # model.set_weights(weights['arr_2'])
# # config = model.get_config()

# model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['sparse_categorical_accuracy'])

# model.fit(training_dataset, epochs=100)

# model.evaluate(testing_dataset)

# model.save("first_trial",save_format='tf')